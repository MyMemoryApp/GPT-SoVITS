import re
import os
import json
import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
from io import BytesIO
from peft import LoraConfig, get_peft_model

from text import chinese
from text import cleaned_text_to_sequence
from text.LangSegmenter import LangSegmenter
from text.cleaner import clean_text

from time import time as ttime
from feature_extractor import cnhubert
from module.mel_processing import mel_spectrogram_torch, spectrogram_torch

from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.module.models import Generator, SynthesizerTrn, SynthesizerTrnV3
from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new

from sv import SV

from transformers import AutoModelForMaskedLM, AutoTokenizer
cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

cnhubert.cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
ssl_model = cnhubert.get_model()

splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}

v3v4set = {"v3", "v4"}

def change_sovits_weights(sovits_path, is_half, prompt_language=None, text_language=None, train_memory=False):
    # dict_language
    vq_model, hps, = None, None

    # version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
    version, model_version, if_lora_v3 = "v2", "v2Pro", False

    # SoVITS_weights_v2Pro/test_e4_s200.pth v2 v2Pro False!
    print(sovits_path, version, model_version, if_lora_v3)

    # dict_language = dict_language_v1 if version == "v1" else dict_language_v2
    # if prompt_language is not None and text_language is not None:
    #     if prompt_language in list(dict_language.keys()):
    #         prompt_text_update, prompt_language_update = (
    #             {"__type__": "update"},
    #             {"__type__": "update", "value": prompt_language},
    #         )
    #     else:
    #         prompt_text_update = {"__type__": "update", "value": ""}
    #         prompt_language_update = {"__type__": "update", "value": i18n("中文")}
    #     if text_language in list(dict_language.keys()):
    #         text_update, text_language_update = {"__type__": "update"}, {"__type__": "update", "value": text_language}
    #     else:
    #         text_update = {"__type__": "update", "value": ""}
    #         text_language_update = {"__type__": "update", "value": i18n("中文")}
    #     if model_version in v3v4set:
    #         visible_sample_steps = True
    #         visible_inp_refs = False
    #     else:
    #         visible_sample_steps = False
    #         visible_inp_refs = True
    #     yield (
    #         {"__type__": "update", "choices": list(dict_language.keys())},
    #         {"__type__": "update", "choices": list(dict_language.keys())},
    #         prompt_text_update,
    #         prompt_language_update,
    #         text_update,
    #         text_language_update,
    #         {
    #             "__type__": "update",
    #             "visible": visible_sample_steps,
    #             "value": 32 if model_version == "v3" else 8,
    #             "choices": [4, 8, 16, 32, 64, 128] if model_version == "v3" else [4, 8, 16, 32],
    #         },
    #         {"__type__": "update", "visible": visible_inp_refs},
    #         {"__type__": "update", "value": False, "interactive": True if model_version not in v3v4set else False},
    #         {"__type__": "update", "visible": True if model_version == "v3" else False},
    #         {"__type__": "update", "value": i18n("模型加载中，请等待"), "interactive": False},
    #     )

    dict_s2 = load_sovits_new(sovits_path, memory=train_memory)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
        hps.model.version = "v2"  # v3model,v2sybomls
    elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"

    version = hps.model.version
    # print("sovits版本:",hps.model.version)
    if model_version not in v3v4set:
        if "Pro" not in model_version:
            model_version = version
        else:
            hps.model.version = model_version
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    else:
        hps.model.version = model_version
        vq_model = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )

    if "pretrained" not in sovits_path:
        try:
            del vq_model.enc_q
        except:
            pass

    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)

    vq_model.eval()
    vq_model.load_state_dict(dict_s2["weight"], strict=False)

    # with open("./weight.json") as f:
    #     data = f.read()
    #     data = json.loads(data)
    #     data["SoVITS"][version] = sovits_path

    # with open("./weight.json", "w") as f:
    #     f.write(json.dumps(data))

    return vq_model, hps

def change_gpt_weights(gpt_path, is_half):
    hz, max_sec, t2s_model, config = None, None, None, None
    hz = 50
    version="v2"
    dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()

    # total = sum([param.nelement() for param in t2s_model.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))

    # with open("./weight.json") as f:
    #     data = f.read()
    #     data = json.loads(data)
    #     data["GPT"][version] = gpt_path

    # with open("./weight.json", "w") as f:
    #     f.write(json.dumps(data))

    return hz, max_sec, t2s_model, config


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

punctuation = set(["!", "?", "…", ",", ".", "-", " "])
def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def process_text(texts):
    _text = []
    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError("请输入有效文本")
    for text in texts:
        if text in [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


def clean_text_inf(text, language, version):
    language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


def get_bert_inf(phones, word2ph, norm_text, language, is_half=False):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)  # .to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


def get_phones_and_bert(text, language, version, is_half=False, final=False):
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "all_zh":
            if re.search(r"[A-Za-z]", formattext):
                formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext, "zh", version)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = get_bert_feature(norm_text, word2ph).to(device)
        elif language == "all_yue" and re.search(r"[A-Za-z]", formattext):
            formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
            formattext = chinese.mix_text_normalize(formattext)
            return get_phones_and_bert(formattext, "yue", version)
        else:
            phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
        textlist = []
        langlist = []
        if language == "auto":
            for tmp in LangSegmenter.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegmenter.getTexts(text):
                if langlist:
                    if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                        textlist[-1] += tmp["text"]
                        continue
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日韩文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, version, final=True)

    dtype = torch.float16 if is_half == True else torch.float32
    return phones, bert.to(dtype), norm_text


resample_transform_dict = {}
def resample(audio_tensor, sr0, sr1, device):
    global resample_transform_dict
    key = "%s-%s-%s" % (sr0, sr1, str(device))
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)


def get_spepc(hps, filename, dtype, device, is_v2pro=False):
    # audio = load_audio(filename, int(hps.data.sampling_rate))

    # audio, sampling_rate = librosa.load(filename, sr=int(hps.data.sampling_rate))
    # audio = torch.FloatTensor(audio)

    sr1 = int(hps.data.sampling_rate)
    audio, sr0 = torchaudio.load(filename)
    if sr0 != sr1:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)
        audio = resample(audio, sr0, sr1, device)
    else:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)

    maxx = audio.abs().max()
    if maxx > 1:
        audio /= min(2, maxx)
    spec = spectrogram_torch(
        audio,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    spec = spec.to(dtype)
    if is_v2pro == True:
        audio = resample(audio, sr1, 16000, device).to(dtype)
    return spec, audio


mel_fn = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1024,
        "win_size": 1024,
        "hop_size": 256,
        "num_mels": 100,
        "sampling_rate": 24000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)
mel_fn_v4 = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1280,
        "win_size": 1280,
        "hop_size": 320,
        "num_mels": 100,
        "sampling_rate": 32000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)

spec_min = -12
spec_max = 2

def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1


def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min

def audio_sr(audio, sr):
    global sr_model
    if sr_model == None:
        from tools.audio_sr import AP_BWE

        try:
            sr_model = AP_BWE(device, DictToAttrRecursive)
        except FileNotFoundError:
            raise Exception("你没有下载超分模型的参数，因此不进行超分。如想超分请先参照教程把文件下载好")
            return audio.cpu().detach().numpy(), sr
    return sr_model(audio, sr)


def get_tts_wav(
    ref_wav_path,
    prompt_text,
    # prompt_language,
    text,
    # text_language,
    # how_to_cut=i18n("不切"),
    output_dir,
    gpt_path,
    sovits_path,
    is_half=False,
    top_k=20,
    top_p=0.6,
    temperature=0.6,
    ref_free=False,
    speed=1,
    if_freeze=False,
    inp_refs=None,
    sample_steps=8,
    if_sr=False,
    pause_second=0.3,
    train_memory=False
):
    # print("get_tts_wav", ref_wav_path, prompt_text, prompt_language, text, text_language)
    # print("get_tts_wav", top_k, top_p, temperature, ref_free, speed, if_freeze, inp_refs, sample_steps, if_sr, pause_second)
    # get_tts_wav /private/var/folders/kt/_smk6hpd3z994h13fxg2q18w0000gn/T/gradio/3e2c21821a60d25b333ca6c56a857eadf90be00652ebde65cd79b5f1e32bfc3a/2.wav.reformatted_vocals.wav_main_vocal.wav_0000000000_0000158080.wav 
    # 情况不妙时这就是我的处理方式丑话说在前头 
    # 中文 
    # 楼下一个男人病得要死，那间壁的一家唱着留声机;对面是弄孩子。楼上有两人狂笑;还有打牌声。人类的悲欢并不相通，我只觉得他们吵闹。 
    # 中文 
    # 凑四句一切
    # get_tts_wav 15 1 1 False 1 False None 8 False 0.3

    top_k = 15
    top_p = 1
    temperature = 1
    ref_free = False
    # speed = 1
    speed = 0.75
    if_freeze = False
    inp_refs = None
    sample_steps = 8
    if_sr = False
    pause_second = 0.3

    cache = {}

    # if ref_wav_path:
    #     pass
    # else:
    #     gr.Warning(i18n("请上传参考音频"))

    # if text:
    #     pass
    # else:
    #     gr.Warning(i18n("请填入推理文本"))

    # todo
    # gpt_path = ""
    # sovits_path = ""
    # is_half = False

    t = []
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True

    version="v2"
    model_version = "v2Pro"
    if_sr = False
    dtype = torch.float16 if is_half == True else torch.float32

    # if model_version in v3v4set:
    #     ref_free = False  # s2v3暂不支持ref_free
    # else:
    #     if_sr = False
    #         if_sr = False

    # if model_version not in {"v3", "v4", "v2Pro", "v2ProPlus"}:
    #     clean_bigvgan_model()
    #     clean_hifigan_model()
    #     clean_sv_cn_model()

    t0 = ttime()

    # i18n("中文"): "all_zh",  # 全部按中文识别
    # i18n("中英混合"): "zh",  # 按中英混合识别####不变
    prompt_language = "zh"
    text_language = "zh"
    # prompt_language = dict_language[prompt_language]
    # text_language = dict_language[text_language]

    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."
        print("实际输入的参考文本:", prompt_text)

    text = text.strip("\n")
    # if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text

    print("实际输入的目标文本:", text)

    vq_model, hps = change_sovits_weights(sovits_path, is_half, train_memory=train_memory)
    if train_memory:
        bio = BytesIO()
        bio.write(gpt_path.data)
        bio.seek(0)
        gpt_path = bio

        wio = BytesIO()
        wio.write(ref_wav_path.data)
        wio.seek(0)
        ref_wav_path = wio
    hz, max_sec, t2s_model, gpt_config = change_gpt_weights(gpt_path, is_half)

    zero_wav = np.zeros(
        int(hps.data.sampling_rate * pause_second),
        dtype=np.float16 if is_half == True else np.float32,
    )
    zero_wav_torch = torch.from_numpy(zero_wav)
    if is_half == True:
        zero_wav_torch = zero_wav_torch.half().to(device)
    else:
        zero_wav_torch = zero_wav_torch.to(device)

    if not ref_free:
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                raise OSError("参考音频在3~10秒范围外,请更换！")
            
            wav16k = torch.from_numpy(wav16k)
            if is_half == True:
                wav16k = wav16k.half().to(device)
            else:
                wav16k = wav16k.to(device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(device)

    t1 = ttime()
    t.append(t1 - t0)

    text = cut1(text)
    # if how_to_cut == i18n("凑四句一切"):
    #     text = cut1(text)
    # elif how_to_cut == i18n("凑50字一切"):
    #     text = cut2(text)
    # elif how_to_cut == i18n("按中文句号。切"):
    #     text = cut3(text)
    # elif how_to_cut == i18n("按英文句号.切"):
    #     text = cut4(text)
    # elif how_to_cut == i18n("按标点符号切"):
    #     text = cut5(text)

    while "\n\n" in text:
        text = text.replace("\n\n", "\n")

    print("实际输入的目标文本(切句后):", text)

    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)

    audio_opt = []

    ###s2v3暂不支持ref_free
    if not ref_free:
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, version)

    for i_text, text in enumerate(texts):
        # 解决输入目标文本的空行导致报错的问题
        if len(text.strip()) == 0:
            continue

        if text[-1] not in splits:
            text += "。" if text_language != "en" else "."

        print("实际输入的目标文本(每句):", text)

        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language, version)
        print("前端处理后的文本(每句):", norm_text2)
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        t2 = ttime()
        # cache_key="%s-%s-%s-%s-%s-%s-%s-%s"%(ref_wav_path,prompt_text,prompt_language,text,text_language,top_k,top_p,temperature)
        # print(cache.keys(),if_freeze)
        if i_text in cache and if_freeze == True:
            pred_semantic = cache[i_text]
        else:
            with torch.no_grad():
                pred_semantic, idx = t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if ref_free else prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=hz * max_sec,
                )
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                cache[i_text] = pred_semantic

        t3 = ttime()
        is_v2pro = model_version in {"v2Pro", "v2ProPlus"}
        # print(23333,is_v2pro,model_version)
        ###v3不存在以下逻辑和inp_refs

        sv_cn_model = SV(device, is_half)
        try:
            torch.cuda.empty_cache()
        except:
            pass

        if model_version not in v3v4set:
            refers = []
            if is_v2pro:
                sv_emb = []
                # if sv_cn_model == None:
                #     init_sv_cn()

            # if inp_refs:
            #     for path in inp_refs:
            #         try:  #####这里加上提取sv的逻辑，要么一堆sv一堆refer，要么单个sv单个refer
            #             refer, audio_tensor = get_spepc(hps, path.name, dtype, device, is_v2pro)
            #             refers.append(refer)
            #             if is_v2pro:
            #                 sv_emb.append(sv_cn_model.compute_embedding3(audio_tensor))
            #         except:
            #             traceback.print_exc()

            if len(refers) == 0:
                if train_memory:
                    ref_wav_path.seek(0)
                    
                refers, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device, is_v2pro)
                refers = [refers]
                if is_v2pro:
                    sv_emb = [sv_cn_model.compute_embedding3(audio_tensor)]

            if is_v2pro:
                audio = vq_model.decode(
                    pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers, speed=speed, sv_emb=sv_emb
                )[0][0]
            else:
                audio = vq_model.decode(
                    pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers, speed=speed
                )[0][0]
        else:
            # print("123112111111111111111111111111")
            pass
            # refer, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device)
            # phoneme_ids0 = torch.LongTensor(phones1).to(device).unsqueeze(0)
            # phoneme_ids1 = torch.LongTensor(phones2).to(device).unsqueeze(0)
            # fea_ref, ge = vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
            # ref_audio, sr = torchaudio.load(ref_wav_path)
            # ref_audio = ref_audio.to(device).float()
            # if ref_audio.shape[0] == 2:
            #     ref_audio = ref_audio.mean(0).unsqueeze(0)
            # tgt_sr = 24000 if model_version == "v3" else 32000
            # if sr != tgt_sr:
            #     ref_audio = resample(ref_audio, sr, tgt_sr, device)
            # # print("ref_audio",ref_audio.abs().mean())
            # mel2 = mel_fn(ref_audio) if model_version == "v3" else mel_fn_v4(ref_audio)
            # mel2 = norm_spec(mel2)
            # T_min = min(mel2.shape[2], fea_ref.shape[2])
            # mel2 = mel2[:, :, :T_min]
            # fea_ref = fea_ref[:, :, :T_min]
            # Tref = 468 if model_version == "v3" else 500
            # Tchunk = 934 if model_version == "v3" else 1000
            # if T_min > Tref:
            #     mel2 = mel2[:, :, -Tref:]
            #     fea_ref = fea_ref[:, :, -Tref:]
            #     T_min = Tref
            # chunk_len = Tchunk - T_min
            # mel2 = mel2.to(dtype)
            # fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge, speed)
            # cfm_resss = []
            # idx = 0
            # while 1:
            #     fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
            #     if fea_todo_chunk.shape[-1] == 0:
            #         break
            #     idx += chunk_len
            #     fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
            #     cfm_res = vq_model.cfm.inference(
            #         fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0
            #     )
            #     cfm_res = cfm_res[:, :, mel2.shape[2] :]
            #     mel2 = cfm_res[:, :, -T_min:]
            #     fea_ref = fea_todo_chunk[:, :, -T_min:]
            #     cfm_resss.append(cfm_res)
            # cfm_res = torch.cat(cfm_resss, 2)
            # cfm_res = denorm_spec(cfm_res)
            # if model_version == "v3":
            #     if bigvgan_model == None:
            #         init_bigvgan()
            # else:  # v4
            #     if hifigan_model == None:
            #         init_hifigan()
            # vocoder_model = bigvgan_model if model_version == "v3" else hifigan_model
            # with torch.inference_mode():
            #     wav_gen = vocoder_model(cfm_res)
            #     audio = wav_gen[0][0]  # .cpu().detach().numpy()
        max_audio = torch.abs(audio).max()  # 简单防止16bit爆音
        if max_audio > 1:
            audio = audio / max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav_torch)  # zero_wav
        t4 = ttime()
        t.extend([t2 - t1, t3 - t2, t4 - t3])
        t1 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3])))
    audio_opt = torch.cat(audio_opt, 0)  # np.concatenate
    if model_version in {"v1", "v2", "v2Pro", "v2ProPlus"}:
        opt_sr = 32000
    elif model_version == "v3":
        opt_sr = 24000
    else:
        opt_sr = 48000  # v4

    if if_sr == True and opt_sr == 24000:
        print("音频超分中")
        audio_opt, opt_sr = audio_sr(audio_opt.unsqueeze(0), opt_sr)
        max_audio = np.abs(audio_opt).max()
        if max_audio > 1:
            audio_opt /= max_audio
    else:
        audio_opt = audio_opt.cpu().detach().numpy()

    if str(output_dir).endswith(".wav"):
        output_file = output_dir
    else:
        output_file = os.path.join(output_dir, "output.wav")

    audio_data = (audio_opt * 32767).astype(np.int16)

    # import scipy.io.wavfile as wavfile

    # output_file1 = output_file + "1"
    # wavfile.write(output_file1, opt_sr, audio_data)
    sf.write(output_file, audio_data, opt_sr, subtype='PCM_16')

    # yield opt_sr, (audio_opt * 32767).astype(np.int16)

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

