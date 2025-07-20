import os
import re
import shutil
from tools.uvr5.webui import run_uvr5
from tools.slice_audio import slice
from tools.asr.funasr_asr import execute_asr
from GPT_SoVITS.prepare_datasets.s1_get_text import run_args as prepare_1_run_args
from GPT_SoVITS.prepare_datasets.s2_get_hubert_wav32k import run_args as prepare_2_1_run_args
from GPT_SoVITS.prepare_datasets.s2_get_sv import run_args as prepare_2_2_run_args
from GPT_SoVITS.prepare_datasets.s3_get_semantic import run_args as prepare_3_run_args
from GPT_SoVITS.s2_train import run_args as s2_train_run_args
from GPT_SoVITS.s1_train import run_args as s1_train_run_args
from GPT_SoVITS.inference_run import get_tts_wav as inference_run_args

uvr5_model_1 = "model_bs_roformer_ep_317_sdr_12.9755"
uvr5_model_2 = "onnx_dereverb_By_FoxJoy"
uvr5_model_3 = "VR-DeEchoAggressive"


def dispose_asr(src, output_dir, device, is_half, name="user"):
    print("[dispose_asr] src {} out {}".format(src, output_dir))
     # 1
    src_uvr5 = splite_voice(device=device, src=src, outpu_dir=output_dir)

    # 2
    src_slice = slice_voice(src_uvr5, output_dir, name)

    # 3
    asr_text_file = asr(name, src_slice, output_dir)

def dispose_train_model(root_dir, is_half, process_flag: int, name="user"):
    train_port = 20000 + process_flag
    src_slice = os.path.join(root_dir, "slice")
    asr_text_file = os.path.join(root_dir, "asr/user.list")
    output_dir = root_dir
    # 4
    prepare_dir = prepare_data(name, asr_text_file, src_slice, output_dir, is_half, 1)

    # 5
    train_dir = train_data(name, prepare_dir, output_dir, train_port, is_half)

    wav_file = ""
    wav_text = ""
    with open(asr_text_file, 'r') as f:
        for line in f:
            content_list = line.split("|")
            if not wav_file:
                wav_file = content_list[0]
                wav_text = content_list[3]
                continue

            if len(content_list[3]) > len(wav_text):
                wav_file = content_list[0]
                wav_text = content_list[3]
                continue

def dispose(src, output_dir, name, device, is_half, dst_text):

    train_port = 20000
    # src = "/Users/phoenix/Downloads/music/ai/2.wav"
    # output_dir = "/Users/phoenix/Documents/project/GPT-SoVITS/dispose_out/"

    # 1
    src_uvr5 = splite_voice(device=device, src=src, outpu_dir=output_dir)

    # 2
    src_slice = slice_voice(src_uvr5, output_dir, name)

    # 3
    asr_text_file = asr(name, src_slice, output_dir)

    # 4
    prepare_dir = prepare_data(name, asr_text_file, src_slice, output_dir, is_half, 1)

    # 5
    train_dir = train_data(name, prepare_dir, output_dir, train_port, is_half)

    wav_file = ""
    wav_text = ""
    with open(asr_text_file, 'r') as f:
        for line in f:
            content_list = line.split("|")
            if not wav_file:
                wav_file = content_list[0]
                wav_text = content_list[3]
                continue

            if len(content_list[3]) > len(wav_text):
                wav_file = content_list[0]
                wav_text = content_list[3]
                continue

    # print("6", wav_file, wav_text)
    # 6 
    inference_result(wav_file, wav_text, dst_text, output_dir, train_dir)

# 分离声源，让个人声音更清晰
def splite_voice(device, src, outpu_dir):
    # model_bs_roformer_ep_317_sdr_12.9755  output/uvr5_opt ['/private/var/folders/kt/_smk6hpd3z994h13fxg2q18w0000gn/T/gradio/2bde8c2ab5ea26a02e8cd128ab9de566fac2288d80c2afedfca937a8b7a7eb32/1.wav'] output/uvr5_opt 10 wav

    outpu_dir_uvr5 = os.path.join(outpu_dir, "urv5")
    os.makedirs(outpu_dir_uvr5, exist_ok=True)

    outpu_dir_uvr5_1 = os.path.join(outpu_dir_uvr5, "1")
    os.makedirs(outpu_dir_uvr5_1, exist_ok=True)

    outpu_dir_uvr5_2 = os.path.join(outpu_dir_uvr5, "2")
    os.makedirs(outpu_dir_uvr5_2, exist_ok=True)

    # outpu_dir_uvr5_3 = os.path.join(outpu_dir_uvr5, "3")
    # os.makedirs(outpu_dir_uvr5_3, exist_ok=True)

    uvr5_2_temp = os.path.join(outpu_dir_uvr5_2, "TEMP")

    # os.path.exists(uvr5_2_temp) and shutil.rmtree(uvr5_2_temp)
    # return outpu_dir_uvr5_2

    is_dir = os.path.isdir(src)
    print("[uvr5] model_1 run.\n")
    run_uvr5(
        dv="cpu",
        half=False,
        model_name=uvr5_model_1,

        # 批量处理文件夹下的文件
        inp_root=src if is_dir else "",

        save_root_vocal=outpu_dir_uvr5_1,

        # 单独的文件
        paths=[
            src
        ]if not is_dir else [],

        save_root_ins="",
        agg=10,
        format0="wav"
    )
    print("[uvr5] model_1 run ok.\n")

    print("[uvr5] model_2 run.\n")
    run_uvr5(
        dv="cpu",
        half=False,
        model_name=uvr5_model_2,

        # 批量处理文件夹下的文件
        inp_root=outpu_dir_uvr5_1,

        save_root_vocal=outpu_dir_uvr5_2,

        # 单独的文件
        paths=[
        ],

        save_root_ins="",
        agg=10,
        format0="wav"
    )
    print("[uvr5] model_2 run ok.\n")

    # return outpu_dir_uvr5_2
    # print("[uvr5] model_3 run.\n")
    # run_uvr5(
    #     dv="cpu",
    #     half=False,
    #     model_name=uvr5_model_3,

    #     # 批量处理文件夹下的文件
    #     inp_root=outpu_dir_uvr5_2,

    #     save_root_vocal=outpu_dir_uvr5_3,

    #     # 单独的文件
    #     paths=[
    #     ],

    #     save_root_ins="",
    #     agg=10,
    #     format0="wav"
    # )

    # print("[uvr5] model_3 run ok.\n")
    os.path.exists(uvr5_2_temp) and shutil.rmtree(uvr5_2_temp)
    return outpu_dir_uvr5_2


# 切割音频、让每段音频不要一个长句子
def slice_voice(src, outpu_dir, name):
    # "/opt/miniconda3/envs/GPTSoVits/bin/python" -s tools/slice_audio.py 
    # "dispose_out/urv5/2/2.wav.reformatted_vocals.wav_main_vocal.wav" 
    # "output/slicer_opt" 
    # -34 4000 300 10 500 0.9 0.25 0 1
    # inp dispose_out/urv5/2/2.wav.reformatted_vocals.wav_main_vocal.wav 
    # opt_root output/slicer_opt 
    # threshold -34 min_length 4000 min_interval 300 hop_size 10 
    # max_sil_kept 500 _max 0.9 alpha 0.25 i_part 0 all_part 1
    outpu_dir_slice = os.path.join(outpu_dir, "slice")
    os.makedirs(outpu_dir, exist_ok=True)

    # return outpu_dir_slice
    slice(
        inp=src, 
        opt_root=outpu_dir_slice, 
        # 音量小于这个值视作静音的备选切割点
        threshold=-34, 
        # 每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值
        min_length=4000, 
        # 最短切割间隔
        min_interval=300, 
        hop_size=10, 
        # 切完后静音最多留多长
        max_sil_kept=500, 
        # 归一化后最大值多少
        _max=0.9, 
        # 混多少比例归一化后音频进来
        alpha=0.25, 
        i_part=0, 
        all_part=1,
        name=name,
        )
    return outpu_dir_slice


# 标注音频内容
def asr(name, src, outpu_dir):
    # "/opt/miniconda3/envs/GPTSoVits/bin/python" 
    # -s tools/asr/funasr_asr.py 
    # -i "/Users/phoenix/Documents/project/GPT-SoVITS/dispose_out/slice" 
    # -o "output/asr_opt" 
    # -s large 
    # -l zh 
    # -p float32

    outpu_dir_asr = os.path.join(outpu_dir, "asr")
    os.makedirs(outpu_dir, exist_ok=True)

    # return os.path.join(outpu_dir_asr, "{}.list".format(name))
    
    # funasr model_size 无效
    execute_asr(
        input_folder=src,
        output_folder=outpu_dir_asr,
        model_size="float32",
        language="zh",
        name=name,
    )

    return os.path.join(outpu_dir_asr, "{}.list".format(name))

# all_part 并行处理的核心数目、 cuda_device gpu 的数目
def prepare_data(name, input_text_file, input_audio_dir, output_dir, is_half, all_parts=2, cuda_device=0):
    output_dir_prepare = os.path.join(output_dir, "prepare")
    os.makedirs(output_dir_prepare, exist_ok=True)

    # return output_dir_prepare
    # 1
    # out 2-file、3-dir
    for i in range(all_parts):
        prepare_1_run_args(
            input_text=input_text_file,
            input_wav_dir=input_audio_dir,
            experiment_name=name,
            output_dir=output_dir_prepare,
            index_part=i,
            count_part=all_parts,
            half=is_half,
            cuda_device=cuda_device
        )
    
    opt = []
    path_text = "%s/2-name2text.txt" % output_dir_prepare
    for i_part in range(all_parts):  # txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
        txt_path = "%s/2-name2text-%s.txt" % (output_dir_prepare, i_part)
        with open(txt_path, "r", encoding="utf8") as f:
            opt += f.read().strip("\n").split("\n")
        os.remove(txt_path)
    with open(path_text, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")

    # 2 -1
    # out 4、5 dir
    for i in range(all_parts):
        prepare_2_1_run_args(
            input_text=input_text_file,
            input_wav_dir=input_audio_dir,
            experiment_name=name,
            output_dir=output_dir_prepare,
            index_part=i,
            count_part=all_parts,
            half=is_half,
            cuda_device=cuda_device
        )

    # 2 -2
    # out 7 dir
    for i in range(all_parts):
        prepare_2_2_run_args(
            input_text=input_text_file,
            input_wav_dir=input_audio_dir,
            experiment_name=name,
            output_dir=output_dir_prepare,
            index_part=i,
            count_part=all_parts,
            half=is_half,
            cuda_device=cuda_device
        )
    
    # 3
    # out 6 dir
    for i in range(all_parts):
        prepare_3_run_args(
            input_text=input_text_file,
            experiment_name=name,
            output_dir=output_dir_prepare,
            index_part=i,
            count_part=all_parts,
            half=is_half,
            cuda_device=cuda_device
        )

    opt = ["item_name\tsemantic_audio"]
    for i_part in range(all_parts):
        semantic_path = "%s/6-name2semantic-%s.tsv" % (output_dir_prepare, i_part)
        with open(semantic_path, "r", encoding="utf8") as f:
            opt += f.read().strip("\n").split("\n")
        os.remove(semantic_path)
    
    path_semantic = "%s/6-name2semantic.tsv" % output_dir_prepare
    with open(path_semantic, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")
    
    return output_dir_prepare


def train_data(name, prepare_dir, output_dir, train_port, is_half):
    output_dir_train = os.path.join(output_dir, "train")
    os.makedirs(output_dir_train, exist_ok=True)
    # return output_dir_train

    # 训练数据 gpt
    s2_train(name, prepare_dir, output_dir_train, train_port, is_half)

    # sovits
    s1_train(name, prepare_dir, output_dir_train, is_half)

    return output_dir_train


def s2_train(name, prepare_dir, output_dir_train, port, is_half):

    logs_s2_v2_pro_dir = os.path.join(prepare_dir, "logs_s2_v2Pro")
    os.makedirs(logs_s2_v2_pro_dir, exist_ok=True)

    s2_train_run_args(
        name=name,
        prepare_dir=prepare_dir,
        output_dir=output_dir_train,
        is_half=is_half,
        port=port
    )


def s1_train(name, prepare_dir, output_dir_train, is_half):
    s1_train_run_args(
        name=name,
        prepare_dir=prepare_dir,
        output_dir=output_dir_train,
        is_half=is_half
    )


gpt_pattern = re.compile(r'.*e(\d+)\.ckpt$')
sovits_pattern = re.compile(r'.*s(\d+)\.pth$')

def get_last_file(dir, kind=1):
    pattern = gpt_pattern
    if 2 == kind:
        pattern = sovits_pattern
    max_num = -1
    max_file = None

    for filename in os.listdir(dir):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                max_file = filename

    return max_file

def inference_result_by_minio(src_wav_path, src_wav_text, dst_text, output_dir, gpt_data, sovits_data, file_name):
    inference_outpu_dir = os.path.join(output_dir, "inference")
    os.makedirs(inference_outpu_dir, exist_ok=True)
    inference_file = os.path.join(inference_outpu_dir, file_name)

    # gpt_path = os.path.join(train_path, "gpt_train")
    # gpt_file = os.path.join(gpt_path, get_last_file(gpt_path))
    # gpt_file = os.path.join(train_path, "train", "user-e15.ckpt")

    # sovits_path= os.path.join(train_path, "sovits_train")
    # sovits_file = os.path.join(sovits_path, get_last_file(sovits_path, 2))
    # sovits_file = os.path.join(train_path, "train", "user_e8_s400.pth")

    inference_run_args(
        src_wav_path,
        src_wav_text,
        dst_text,
        inference_file,
        gpt_data,
        sovits_data,
        train_memory=True
    )

def inference_result(src_wav_path, src_wav_text, dst_text, output_dir, train_path):
    inference_outpu_dir = os.path.join(output_dir, "inference")
    os.makedirs(inference_outpu_dir, exist_ok=True)

    gpt_path = os.path.join(train_path, "gpt_train")
    gpt_file = os.path.join(gpt_path, get_last_file(gpt_path))

    sovits_path= os.path.join(train_path, "sovits_train")
    sovits_file = os.path.join(sovits_path, get_last_file(sovits_path, 2))

    inference_run_args(
        src_wav_path,
        src_wav_text,
        dst_text,
        inference_outpu_dir,
        gpt_file,
        sovits_file,
    )

def generate_with_inference(name, dst_text, output_dir):
    output_dir = os.path.join(output_dir, name)
    output_dir_train = os.path.join(output_dir, "train")
    asr_text_file = os.path.join(output_dir, "asr", "{}.list".format(name))
    wav_file = ""
    wav_text = ""
    with open(asr_text_file, 'r') as f:
        for line in f:
            content_list = line.split("|")
            if not wav_file:
                wav_file = content_list[0]
                wav_text = content_list[3]
                continue

            if len(content_list[3]) > len(wav_text):
                wav_file = content_list[0]
                wav_text = content_list[3]
                continue

    inference_result(wav_file, wav_text, dst_text, output_dir, output_dir_train)         

if __name__ == "__main__":
    name = "xia"
    # dst_text = "楼下一个男人病得要死，那间壁的一家唱着留声机;对面是弄孩子。楼上有两人狂笑;还有打牌声。人类的悲欢并不相通，我只觉得他们吵闹。"
    dst_text = "生与死轮回不止，我们生，他们死"
    # src = "/Users/phoenix/Downloads/music/ai/2.wav"
    # output_dir = "/Users/phoenix/Documents/project/GPT-SoVITS/dispose_out/"

    # name = "luo1"
    # src = "/Users/phoenix/Downloads/music/ai_1/luo_1.wav"
    # output_dir = "/Users/phoenix/Documents/project/GPT-SoVITS/1_output/luo1"
    # dispose(src, output_dir, name, "cpu", False, dst_text)

    name = "luo1"
    output_dir = "/Users/phoenix/Documents/project/GPT-SoVITS/1_output/"
    dst_text = "楼下一个男人病得要死，那间壁的一家唱着留声机;对面是弄孩子。楼上有两人狂笑;还有打牌声。人类的悲欢并不相通，我只觉得他们吵闹。"
    generate_with_inference(name, dst_text, output_dir)