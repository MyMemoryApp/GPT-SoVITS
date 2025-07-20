import io
import os
import time
import redis
import logging
import traceback
import urllib3

import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
from minio import Minio
from minio.error import S3Error
from datetime import timedelta
from typing import Optional, List, Dict, Any, Union

import dispose as voice_clone

minio_data_path = "/private/var/memory_chat/minio"

# 配置日志，方便调试
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

urllib3_logger = logging.getLogger('urllib3')
urllib3_logger.setLevel(logging.CRITICAL)

vc_logger = logging.getLogger('voice_clone')
vc_logger.setLevel(logging.INFO)
vc_logger.disabled
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

vc_logger.addHandler(ch)
vc_logger.propagate = False

RedisStream      = "voice_clone_task"
RedisStreamGroup = "voice_clone_task_g"
VoiceCloneTableName = "voice_clone"
InferenceOrderTableName = "inference_orders"
class PostgresDB:
    """
    一个用于操作 PostgreSQL 数据库的简单封装类。
    - 使用连接池管理连接。
    - 提供查询和更新方法。
    - 使用上下文管理器确保连接的正确释放。
    """
    _connection_pool = None

    _get_voice_clone_info_sql = "SELECT id, status FROM {name} WHERE id = %s".format(name=VoiceCloneTableName)
    _update_voice_clone_info_sql = "UPDATE {name} SET status = %s WHERE id = %s".format(name=VoiceCloneTableName)
    _get_inferece_order_sql = """
        SELECT vid, date_str, COALESCE(datas->>%s, '') AS data
        FROM {name}
        WHERE id = %s;
    """.format(name=InferenceOrderTableName)

    def __init__(self, db_params, min_conn=1, max_conn=10):
        """
        初始化数据库连接池。

        Args:
            db_params (dict): 数据库连接参数，例如：
                              {
                                  "database": "your_db",
                                  "user": "your_user",
                                  "password": "your_password",
                                  "host": "localhost",
                                  "port": "5432"
                              }
            min_conn (int): 连接池中保持的最小连接数。
            max_conn (int): 连接池允许的最大连接数。
        """
        if not PostgresDB._connection_pool:
            try:
                PostgresDB._connection_pool = pool.SimpleConnectionPool(
                    min_conn,
                    max_conn,
                    **db_params
                )
                logging.info(f"数据库连接池创建成功 (min: {min_conn}, max: {max_conn})")
            except Exception as e:
                logging.error(f"无法连接到 PostgreSQL 数据库: {e}")
                raise e

    @contextmanager
    def get_connection(self):
        """
        使用上下文管理器从连接池获取一个连接。
        使用 'with' 语句可以自动处理连接的获取和释放。
        """
        conn = None
        if not self._connection_pool:
            raise ConnectionError("数据库连接池尚未初始化。")
        try:
            conn = self._connection_pool.getconn()
            yield conn
        except Exception as e:
            logging.error(f"获取数据库连接失败: {e}")
            raise
        finally:
            if conn:
                self._connection_pool.putconn(conn)

    def query(self, sql, params=None, fetch_one=False):
        """
        执行 SELECT 查询。

        Args:
            sql (str): 要执行的 SQL 语句 (使用 %s 作为占位符)。
            params (tuple, optional): 查询参数。默认为 None。
            fetch_one (bool, optional): 是否只获取第一条记录。默认为 False (获取所有记录)。

        Returns:
            list or None: 查询结果。
                                  - fetch_one=False: 返回一个列表。
                                  - fetch_one=True: 返回单个列表。
                                  - 如果没有结果或发生错误，返回 None。
        """
        with self.get_connection() as conn:
            # 使用 'with conn.cursor()' 确保游标被关闭
            with conn.cursor() as cur:
                try:
                    cur.execute(sql, params)
                    if fetch_one:
                        result = cur.fetchone()
                        # print(result)
                        return result
                    else:
                        results = cur.fetchall()
                        # return [dict(row) for row in results]
                        return results
                except Exception as e:
                    logging.error(f"查询执行失败: {e}\nSQL: {sql}\nParams: {params}")
                    return None

    def update(self, sql, params=None):
        """
        执行 INSERT, UPDATE, DELETE 等更新操作。

        Args:
            sql (str): 要执行的 SQL 语句 (使用 %s 作为占位符)。
            params (tuple, optional): 操作参数。默认为 None。

        Returns:
            int or None: 受影响的行数。如果发生错误，返回 None。
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(sql, params)
                    affected_rows = cur.rowcount
                    conn.commit()  # 提交事务
                    logging.info(f"更新操作成功，{affected_rows} 行受到影响。")
                    return affected_rows
                except psycopg2.Error as e:
                    conn.rollback()  # 如果发生错误，回滚事务
                    logging.error(f"更新执行失败: {e}\nSQL: {sql}\nParams: {params}")
                    return None

    def close_pool(self):
        """
        关闭数据库连接池。
        """
        if PostgresDB._connection_pool:
            PostgresDB._connection_pool.closeall()
            PostgresDB._connection_pool = None
            logging.info("数据库连接池已关闭。")
    
    def get_voice_clone_info_status(self, uid: str):
        # id status
        res = self.query(PostgresDB._get_voice_clone_info_sql, (uid, ), fetch_one=True)
        if not res:
            return -1
        
        return res[1]
    
    def update_voice_clone_info_status(self, uid: str, status: int):
        res = self.update(PostgresDB._update_voice_clone_info_sql, (status, uid))

    def get_inference_order_by_timestr(self, order_id: str, time_str: str):
        res = self.query(PostgresDB._get_inferece_order_sql, (time_str, order_id), fetch_one=True)
        return res


class MinioClient:
    """
    一个用于操作 MinIO (或任何S3兼容对象存储) 的简单封装类。
    """
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False):
        """
        初始化 MinIO 客户端。

        Args:
            endpoint (str): MinIO 服务器的地址和端口，例如 "localhost:9000"。
            access_key (str): 访问密钥 (Access Key)。
            secret_key (str): 私有密钥 (Secret Key)。
            secure (bool): 是否使用 HTTPS。对于本地 Docker 环境，通常为 False。
        """
        self.client = None
        try:
            self.client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure
            )
            logging.info(f"成功连接到 MinIO 服务 at {endpoint}")
        except (S3Error, TypeError) as e:
            logging.error(f"无法连接到 MinIO: {e}")
            raise

    def create_bucket_if_not_exists(self, bucket_name: str) -> bool:
        """
        如果存储桶不存在，则创建它。

        Args:
            bucket_name (str): 存储桶的名称。

        Returns:
            bool: 如果成功创建或已存在，返回 True；否则返回 False。
        """
        try:
            found = self.client.bucket_exists(bucket_name)
            if not found:
                self.client.make_bucket(bucket_name)
                logging.info(f"存储桶 '{bucket_name}' 创建成功。")
            else:
                logging.info(f"存储桶 '{bucket_name}' 已存在。")
            return True
        except S3Error as e:
            logging.error(f"操作存储桶 '{bucket_name}' 时发生错误: {e}")
            return False

    def upload_file(self, bucket_name: str, object_name: str, file_path: str) -> Optional[str]:
        """
        从本地文件路径上传文件。

        Args:
            bucket_name (str): 目标存储桶名称。
            object_name (str): 在存储桶中保存的对象名称 (例如 "images/my-photo.jpg")。
            file_path (str): 本地文件的完整路径。

        Returns:
            Optional[str]: 成功则返回对象的 ETag，失败则返回 None。
        """
        try:
            result = self.client.fput_object(bucket_name, object_name, file_path)
            logging.info(f"文件 '{file_path}' 已成功上传为 '{object_name}' 到存储桶 '{bucket_name}'。")
            return result.etag
        except S3Error as e:
            logging.error(f"上传文件 '{file_path}' 失败: {e}")
            return None
            
    def upload_data(self, bucket_name: str, object_name: str, data: bytes) -> Optional[str]:
        """
        上传内存中的二进制数据。

        Args:
            bucket_name (str): 目标存储桶名称。
            object_name (str): 在存储桶中保存的对象名称。
            data (bytes): 要上传的二进制数据。

        Returns:
            Optional[str]: 成功则返回对象的 ETag，失败则返回 None。
        """
        try:
            result = self.client.put_object(
                bucket_name, 
                object_name, 
                io.BytesIO(data), 
                len(data)
            )
            logging.info(f"数据已成功上传为 '{object_name}' 到存储桶 '{bucket_name}'。")
            return result.etag
        except S3Error as e:
            logging.error(f"上传数据为 '{object_name}' 失败: {e}")
            return None


    def download_file(self, bucket_name: str, object_name: str, file_path: str) -> bool:
        """
        将对象下载到本地文件。

        Args:
            bucket_name (str): 所在存储桶的名称。
            object_name (str): 要下载的对象名称。
            file_path (str): 保存到的本地文件路径。

        Returns:
            bool: 成功返回 True，失败返回 False。
        """
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
            logging.info(f"对象 '{object_name}' 已成功下载到 '{file_path}'。")
            return True
        except S3Error as e:
            logging.error(f"下载对象 '{object_name}' 失败: {e}")
            return False
    
    def get_file(self, bucket_name: str, object_name: str) -> urllib3.BaseHTTPResponse:
        resp = self.client.get_object(bucket_name, object_name)
        return resp
        

    def get_presigned_url(self, bucket_name: str, object_name: str, expires_days: int = 7) -> Optional[str]:
        """
        为对象生成一个临时的、可公开访问的下载链接。

        Args:
            bucket_name (str): 所在存储桶的名称。
            object_name (str): 对象名称。
            expires_days (int): 链接的有效天数，默认为 7 天。

        Returns:
            Optional[str]: 生成的 URL 字符串，如果失败则返回 None。
        """
        try:
            url = self.client.get_presigned_url(
                "GET",
                bucket_name,
                object_name,
                expires=timedelta(days=expires_days),
            )
            logging.info(f"为对象 '{object_name}' 生成了有效期 {expires_days} 天的分享链接。")
            return url
        except S3Error as e:
            logging.error(f"生成分享链接失败: {e}")
            return None

    def list_objects(self, bucket_name: str, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出存储桶中的对象。

        Args:
            bucket_name (str): 存储桶名称。
            prefix (Optional[str]): (可选) 只列出有指定前缀的对象。

        Returns:
            List[Dict[str, Any]]: 包含对象信息的字典列表 (name, size, last_modified)。
        """
        try:
            objects = self.client.list_objects(bucket_name, prefix=prefix, recursive=True)
            object_list = [
                {
                    "name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified
                }
                for obj in objects
            ]
            return object_list
        except S3Error as e:
            logging.error(f"列出存储桶 '{bucket_name}' 中的对象失败: {e}")
            return []
    
    def get_bucket_location_files(self, bucket_name: str, prefix: str) -> List[urllib3.response.HTTPResponse]:
        files = self.list_objects(bucket_name, prefix)
        if not files:
            return []
        
        reslut = []
        for f in files:
            name = f.get("name")
            if not name:
                continue
            
            res = self.get_file(bucket_name, name)
            reslut.append(res)
            res.data

        return reslut
        
    def download_bucket_location_files(self, bucket_name: str, prefix: str, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        files = self.list_objects(bucket_name, prefix)
        if not files:
            return
        
        for f in files:
            name = f.get("name")
            if not name:
                continue
            
            ok = self.download_file(bucket_name, name, os.path.join(save_path, name))
            if not ok:
                raise Exception("[minio] download_file fail. {} {}".format(bucket_name, name))

    def delete_object(self, bucket_name: str, object_name: str) -> bool:
        """
        删除一个对象。

        Args:
            bucket_name (str): 存储桶名称。
            object_name (str): 要删除的对象名称。

        Returns:
            bool: 成功返回 True，失败返回 False。
        """
        try:
            self.client.remove_object(bucket_name, object_name)
            logging.info(f"对象 '{object_name}' 已从存储桶 '{bucket_name}' 中删除。")
            return True
        except S3Error as e:
            logging.error(f"删除对象 '{object_name}' 失败: {e}")
            return False


class RedisClient:
    # 定义两个 Lua 脚本
    get_current_lua = """
    local uid = ARGV[1]
    local controller = KEYS[1]

    local list_key = redis.call("HGET", controller, uid.."-key")
    local offset = redis.call("HGET", controller, uid.."-offset")

    if not list_key or not offset then
        return nil
    end

    offset = tonumber(offset)

    local data = redis.call("LINDEX", list_key, offset)
    return data
    """

    get_next_lua = """
    local date_count = KEYS[1]
    local controller = KEYS[2]
    local uid = ARGV[1]

    local latest_key = redis.call("HGET", controller, "latest-key")
    local latest_offset = redis.call("HGET", controller, "latest-offset")

    if not latest_key or not latest_offset then
        latest_key = redis.call("LINDEX", date_count, 0)
        if not latest_key then
            return nil
        end
        latest_offset = -1
    end

    latest_offset = tonumber(latest_offset)

    local len = redis.call("LLEN", latest_key)
    if latest_offset + 1 < len then
        latest_offset = latest_offset + 1
        redis.call("HSET", controller, uid.."-key", latest_key)
        redis.call("HSET", controller, uid.."-offset", latest_offset)
        redis.call("HSET", controller, "latest-key", latest_key)
        redis.call("HSET", controller, "latest-offset", latest_offset)
        return redis.call("LINDEX", latest_key, latest_offset)
    else
        local idx = redis.call("LRANGE", date_count, 0, -1)
        local next_index = nil
        for i = 1, #idx do
            if idx[i] == latest_key and i < #idx then
                next_index = i
                break
            end
        end

        if not next_index then
            return nil
        end

        local next_key = idx[next_index+1]
        latest_key = next_key
        latest_offset = 0

        redis.call("HSET", controller, uid.."-key", latest_key)
        redis.call("HSET", controller, uid.."-offset", latest_offset)
        redis.call("HSET", controller, "latest-key", latest_key)
        redis.call("HSET", controller, "latest-offset", latest_offset)

        if next_index+1 >= 4 then
            local first_key = idx[1]
            local busy = false
            local keys = redis.call("HKEYS", controller)
            for j = 1, #keys do
                if string.find(keys[j], "-key") then
                    local v = redis.call("HGET", controller, keys[j])
                    if v == first_key then
                        busy = true
                        break
                    end
                end
            end

            if not busy then
                redis.call("LPOP", date_count)
            end
        end

        return redis.call("LINDEX", latest_key, latest_offset)
    end
    """

    def __init__(self, host='localhost', port=6379, db=0, password=None, decode_responses=True):
        """
        初始化 Redis 连接
        :param host: Redis 服务器地址
        :param port: 端口号
        :param db: 数据库号
        :param password: 密码
        :param decode_responses: 是否解码成字符串
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.decode_responses = decode_responses
        self.client = None

        self.__get_current_script = None
        self.__get_next_script = None

        self._connect()

        self.__init = False
        self.__consumer = "consumer-{}".format(process_flag)
        self.__current_task = None

    def _connect(self):
        """创建连接"""
        self.client = redis.StrictRedis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=self.decode_responses
        )


        # 注册脚本
        # self.__get_current_script = self.client.register_script(RedisClient.get_current_lua)
        # self.__get_next_script = self.client.register_script(RedisClient.get_next_lua)

# # 示例 uid
# uid = "uid123"

# # 获取当前数据
# current_data = get_current_script(keys=["list_controller"], args=[uid])
# print(f"当前数据: {current_data}")

# # 获取下一个任务
# next_data = get_next_script(keys=["date_count", "list_controller"], args=[uid])
# print(f"下一个数据: {next_data}")


    def get_client(self):
        """返回 Redis 客户端"""
        return self.client
    
    # def get_current_task(self, uid) -> str:
    #     self.client.ping()

    #     current_data = self.__get_current_script(keys=["list_controller"], args=[uid])
    #     print(f"当前数据: {current_data}")
    #     return str(current_data)
    
    # def get_next_task(self, uid) -> str:
    #     next_data = self.__get_next_script(keys=["date_count", "list_controller"], args=[uid])
    #     print(f"下一个数据: {next_data}")
    #     return next_data

    def close(self):
        """关闭连接（如果用的是连接池可以扩展这里）"""
        # redis.StrictRedis 没有 close()，一般用在连接池场景。
        pass

    def get_task(self):
        from1 = False
        from2 = False
        if not self.__init:
            self.get_stream_group_current()
            from1 = True
        
        if not self.__current_task:
            self.get_stream_group_next()
            from2 = True
        
        vc_logger.info("[get_task] from1 {} from2 {}".format(from1, from2))
        return self.__current_task
    
    def get_stream_group_current(self):
        resp = self.client.xreadgroup(
                RedisStreamGroup,
                self.__consumer,
                streams={
                    RedisStream: "0"
                },
                count=1,
                # 1 min
                # block=1000*60
            )
        
        self.__init = True
        self.__current_task = None
        for msg in resp:
            # ['voice_clone_task', [('1752475453627-0', {'id': 'test-id-1', 'step': '1'})]]
            stream_name = msg[0]
            if not msg[1]:
                return
            
            datas = msg[1][0]
            self.__current_task = datas
        

    def get_stream_group_next(self):
    # XREADGROUP GROUP order-processors worker-1 COUNT 1 BLOCK 0 STREAMS orders >
        resp = self.client.xreadgroup(
            RedisStreamGroup,
            # "consumer-{}".format(process_flag),
            self.__consumer,
            streams={
                RedisStream: ">"
            },
            count=1,
            # 1 min
            block=1000*60
        )
        
        self.__current_task = None
        for msg in resp:
            # ['voice_clone_task', [('1752475453627-0', {'id': 'test-id-1', 'step': '1'})]]
            stream_name = msg[0]
            datas = msg[1][0]
            if not msg[1]:
                return

            self.__current_task = datas
        

    def commit_task(self):
        if not self.__current_task:
            return
        
        msg_id = self.__current_task[0]
        self.client.xack(RedisStream, RedisStreamGroup, msg_id)
        logging.info("commit_task ok ", msg_id)
    
    def get_current(self):
        return self.__current_task

process_flag = 0
redis_client = RedisClient(host='127.0.0.1', port=6379, db=0)
data_dir = "./dispose_datas"

class VoiceCloneStatus():
    fail = -1
    init = 0
    createAsr = 1
    correctAsr = 2
    done = 3

def dispose(db: PostgresDB, redis_cli: RedisClient, minio_cli: MinioClient):
    vc_logger.info("start dispose...")
    has_task = True
    vid = ""
    task_id = ""
    while True:
        try:
            while True:
                has_task = True
                task = redis_cli.get_task()
                if not task:
                    has_task = False
                    break
                
                # wait postgres commit
                time.sleep(10)
                task_id = task[0]
                task_data = task[1]

                vid = task_data.get('id')
                step = task_data.get('step')
                # infrence_data = task_args[2]

                vc_logger.info(" task {} vid {} step {} start".format(task_id, vid, step))

                # step 2 的 id 不是 vid 是 order id
                if "2" != step:
                    task_status = db.get_voice_clone_info_status(vid)
                    if VoiceCloneStatus.fail == task_status:
                        vc_logger.error("db.get_voice_clone_info_status is -1 {}".format(vid))
                        break
                
                root_dir ="{}/{}".format(data_dir, vid)

                #  create asr file
                if "0" == step:
                    if task_status != VoiceCloneStatus.init:
                        vc_logger.error("db.get_voice_clone_info_status task_status != 0 init {} {}".format(vid, task_status))
                        break

                    # minio_cli.get_bucket_location_files(uid, "src")
                    
                    minio_cli.download_bucket_location_files(vid, "src", root_dir)
                    voice_clone.dispose_asr(os.path.join(root_dir, "src"), root_dir, device="cpu", is_half=False)
                    ok = minio_cli.upload_file(vid, "asr/user.list", os.path.join(root_dir, "asr/user.list"))
                    if not ok:
                        vc_logger.error("minio_cli.upload_file fail. {}".format(vid))
                        break
                    
                    db.update_voice_clone_info_status(vid, VoiceCloneStatus.createAsr)
                    redis_cli.commit_task()
                    vc_logger.info("task {} vid {} step {} end".format(task_id, vid, step))
                    break
                
                # create model
                if "1" == step:
                    if task_status != VoiceCloneStatus.correctAsr:
                        vc_logger.error("db.get_voice_clone_info_status task_status != 2 correctAsr {} {}".format(vid, task_status))
                        break

                    # ok = minio_cli.download_file(uid, "asr/user1.list", os.path.join(root_dir, "asr/user.list"))
                    # if not ok:
                    #     raise Exception("minio_cli.download_file vid {}".format(uid))
                    # voice_clone.dispose_train_model(root_dir, False, process_flag)

                    ok = minio_cli.upload_file(vid, "train/user-e15.ckpt", os.path.join(root_dir, "train/gpt_train/user-e15.ckpt"))
                    if not ok:
                        vc_logger.error("minio_cli.upload_file train gpt fail {}".format(vid))
                        break
                    
                    ok = minio_cli.upload_file(vid, "train/user_e8_s400.pth", os.path.join(root_dir, "train/sovits_train/user_e8_s400.pth"))
                    if not ok:
                        vc_logger.error("minio_cli.upload_file train sovits fail {}".format(vid))
                        break

                    db.update_voice_clone_info_status(vid, VoiceCloneStatus.done)
                    redis_cli.commit_task()
                    break
                
                #  inference 
                if "2" == step:
                    order_id = vid
                    time_str = task_data.get("time_str")
                    if not time_str:
                        vc_logger.error("step 2 get time_str fail task {} {}".format(task_id, vid))
                        redis_cli.commit_task()
                        break

                    info = db.get_inference_order_by_timestr(order_id, time_str)
                    if not info or not info[1]:
                        vc_logger.error("step 2 get_inference_order_data fail task {} {}".format(task_id, vid))
                        redis_cli.commit_task()
                        break
                    
                    voice_id = info[0]
                    date_str = info[1]
                    data = info[2]

                    task_status = db.get_voice_clone_info_status(voice_id)
                    if VoiceCloneStatus.done != task_status:
                        vc_logger.error("step 2 db.get_voice_clone_info_status not done {} {}".format(task_status, voice_id))
                        redis_cli.commit_task()
                        break

                    # voice_clone_minio_dir = os.path.join(minio_data_path, voice_id)
                    # asr_text_file = os.path.join(voice_clone_minio_dir, "asr", "user1.list")
                    resp = minio_cli.get_file(voice_id, "asr/user1.list")
                    if not resp:
                        vc_logger.error("step 2  minio_cli.get_file asr/user1.list fail.".format(task_status, voice_id))
                        redis_cli.commit_task()
                        break

                    asr_data = resp.data.decode()
                    # print(asr_data)
                    # time.sleep(60*10)

                    wav_file = ""
                    wav_text = ""
                    for line in asr_data.split("\n"):
                            content_list = line.split("|")
                            if not wav_file:
                                wav_file = content_list[0]
                                wav_text = content_list[3]
                                continue

                            if len(content_list[3]) > len(wav_text):
                                wav_file = content_list[0]
                                wav_text = content_list[3]
                                continue
                            
                    wav_file = wav_file.split("/")[-1]
                    src_file = minio_cli.list_objects(voice_id, "src")
                    for file_info in src_file:
                        file_name: str = file_info.get('name')
                        if wav_file.startswith(file_name.lstrip('src/')):
                            wav_file = file_name
                            break
                    
                    resp = minio_cli.get_file(voice_id, wav_file)
                    if not resp:
                        vc_logger.error("step 2  minio_cli.get_file train/user-e15.ckpt fail.".format(task_status, voice_id))
                        redis_cli.commit_task()
                        break
                    wav_data = resp

                    # time.sleep(60*10)

                    resp = minio_cli.get_file(voice_id, "train/user-e15.ckpt")
                    if not resp:
                        vc_logger.error("step 2  minio_cli.get_file train/user-e15.ckpt fail.".format(task_status, voice_id))
                        redis_cli.commit_task()
                        break
                    gpt_data = resp

                    resp = minio_cli.get_file(voice_id, "train/user_e8_s400.pth")
                    if not resp:
                        vc_logger.error("step 2  minio_cli.get_file train/user_e8_s400.pth fail.".format(task_status, voice_id))
                        redis_cli.commit_task()
                        break
                    sovites_data = resp

                    vc_logger.info("task {} uid {} step {} inference run.".format(task_id, vid, step))
                    voice_clone.inference_result_by_minio(wav_data, wav_text, data, root_dir, gpt_data, sovites_data, "{}{}.wav".format(date_str, time_str))
                    vc_logger.info("task {} uid {} step {} inference done.".format(task_id, vid, step))
                    break
        except Exception as e:
            err = traceback.format_exc().replace("\n", "-->")
            vc_logger.error(err)
        finally:
            if has_task:
                vc_logger.info("task {} uid {} done.".format(task_id, vid))
            else:
                vc_logger.info("wait task...")

            # get task sleep
            time.sleep(10)
            

def main():
    # 1. 配置你的数据库连接参数
    db_connection_params = {
        "host": "localhost",
        "port": "5432",
        "database": "memory",   # 替换成你的数据库名
        "user": "myuser",      # 替换成你的用户名
        "password": "mypassword" # 替换成你的密码
    }


    try:
        db = PostgresDB(db_params=db_connection_params)
        vc_logger.info("init postgres ok.")

        redis_client = RedisClient(host='127.0.0.1', port=6379, db=0)
        vc_logger.info("init redis ok.")

        minio_client = MinioClient(
            endpoint="127.0.0.1:9000",
            access_key="minioadmin",
            secret_key="minioadmin123",
            secure=False  # 本地 http 环境设为 False
        )
        vc_logger.info("init minio ok.")

        dispose(db, redis_client, minio_client)
    except Exception as e:
        logging.error("init db tools fail", traceback.format_exc())

def run_code():
    minio_client = MinioClient(
            endpoint="127.0.0.1:9000",
            access_key="minioadmin",
            secret_key="minioadmin123",
            secure=False  # 本地 http 环境设为 False
        )
    
    
    # res = minio_client.get_bucket_location_files("minio-t1", "src")
    # print("res len is ", len(res))
    vid = "unique-1"
    root_dir = "{}/{}".format(data_dir, vid)
    # minio_client.download_bucket_location_files("unique-1", "src", "{}/unique-1".format(data_dir, ))
    # voice_clone.dispose_asr(os.path.join(root_dir, "src"), root_dir, device="cpu", is_half=False)
    # ok = minio_client.upload_file(vid, "asr/user.list", os.path.join(root_dir, "asr/user.list"))
    # print(ok)

    # voice_clone.dispose_train_model(root_dir, False, process_flag)

    # ref_wav_path = "/Users/phoenix/Documents/project/GPT-SoVITS/dispose_datas/unique-1/slice/user_0001261120_0001431040.wav"
    # ref_text = "这里呀时机来了，我们告诉诺手一声，你这个大招得在有血怒的时候放伤害才高。"
    # text = "啊，没错，上单五虎太完美了，他们最薄弱的地方就只能是召唤师本身了。打崩诺手有点难，只有直接攻击召唤师，那他本人都不想玩了啊，才能真正的打败诺手。"
    # voice_clone.inference_run_args(
    #     ref_wav_path,
    #     ref_text,
    #     text,
    #     output_dir=os.path.join(root_dir, "out/"),
    #     gpt_path = os.path.join(root_dir, "train/gpt_train/user-e10.ckpt"),
    #     sovits_path = os.path.join(root_dir, "train/sovits_train/user_e8_s400.pth"),
    #     speed=1.4,
    # )

#  pip install redis
# pip install psycopg2-binary
# pip install minio


# 使用示例
if __name__ == '__main__':
    # redis_client = RedisClient(host='127.0.0.1', port=6379, db=0)
    # redis_client.get_task()
    # print(redis_client.get_current())
    # redis_client.commit_task()

    # redis_client.get_current_task(uid="a")
    # redis_client.get_next_task(uid="a")
    # run_code()

    minio_cli = MinioClient(
            endpoint="127.0.0.1:9000",
            access_key="minioadmin",
            secret_key="minioadmin123",
            secure=False  # 本地 http 环境设为 False
        )

    # resp = minio_cli.get_file("2db06b0b-6da7-4951-9d55-614fa4378afc", "asr/user1.list")
    # print(resp.data.decode())

    main()
    # root_dir = os.path.join(data_dir, "0f33c79b-0cb6-4f55-b0c2-8624a22cc6de")
    # voice_clone.slice_voice(os.path.join(root_dir, "src"), root_dir, "user")
    # voice_clone.asr("user", os.path.join(root_dir, "slice"), root_dir)

    # print(os.getenv("DIR"))
    # vid = "2db06b0b-6da7-4951-9d55-614fa4378afc"
    # root_dir ="{}/{}".format(data_dir, vid)
    # voice_clone.dispose_asr(os.path.join(root_dir, "src"), root_dir, device="cpu", is_half=False)
    # ok = minio_cli.upload_file(vid, "asr/user.list", os.path.join(root_dir, "asr/user.list"))

    # voice_clone.dispose_train_model(root_dir, False, process_flag)
    
    # minio_cli.upload_file(vid, "train/user-e15.ckpt", os.path.join(root_dir, "train/gpt_train/user-e15.ckpt"))
    # minio_cli.upload_file(vid, "train/user_e8_s400.pth", os.path.join(root_dir, "train/sovits_train/user_e8_s400.pth"))
    
    # 1. 配置你的数据库连接参数
    # db_connection_params = {
    #     "host": "localhost",
    #     "port": "5432",
    #     "database": "memory",   # 替换成你的数据库名
    #     "user": "myuser",      # 替换成你的用户名
    #     "password": "mypassword" # 替换成你的密码
    # }



    # db = PostgresDB(db_params=db_connection_params)
    # print(db.get_voice_clone_info_status("dd329d23-f3e9-408b-8bd7-16a352149b45"))
    # print("ok")