from DataLoader.utils import *
from zipfile import ZipFile, is_zipfile
from tqdm.auto import tqdm
    
SEED = 42



class CICIDS2017():
  
  def __init__(base_self, seed = 0xDEADBEEF, print_able = True, save_csv = True) -> None:
    np.random.seed(seed)
    np.set_printoptions(suppress=True)
    base_self.__PRINT_ABLE = print_able
    base_self.__SAVE_CSV = save_csv
    base_self.__data_df = pd.DataFrame()


    base_self.__Real_cnt = {'BENIGN': 2273097, 'Hulk': 231073, 'PortScan': 158930, 'DDoS': 128027, 'GoldenEye': 10293,
                            'FTP-Patator': 7938, 'SSH-Patator': 5897, 'slowloris': 5796, 'Slowhttptest': 5499, 'Bot': 1966,
                            'Brute-Force': 1507, 'XSS': 652, 'Infiltration': 36, 'Sql-Injection': 21, 'Heartbleed': 11
    }

    base_self.__Label_map = {
      'DoS Hulk':                    'Hulk',
      'DoS GoldenEye':               'GoldenEye',
      'DoS slowloris':               'slowloris',
      'DoS Slowhttptest':            'Slowhttptest',
      'Web Attack � Brute Force':   'Brute-Force',
      'Web Attack � XSS':           'XSS',
      'Web Attack � Sql Injection': 'Sql-Injection',
    }

    base_self.__Category_map = {  
      'BENIGN':        0,
      'DoS Hulk':          1,
      'DDoS':          1,
      'DoS GoldenEye':     1,
      'DoS slowloris':     1,
      'DoS Slowhttptest':  1,
      'PortScan':      2,
      'FTP-Patator':   3,
      'SSH-Patator':   3,
      'Bot':           4,
      'Web Attack � Brute Force':   5,
      'Web Attack � XSS':           5,
      'Web Attack � Sql Injection': 5,
      'Infiltration':  6,
      'Heartbleed':    7
    }

    base_self.__Label_true_name = [  'BENIGN','Hulk', 'PortScan','DDoS','GoldenEye','FTP-Patator','SSH-Patator',
      'slowloris','Slowhttptest','Bot','Brute-Force','XSS','Infiltration',
      'Sql-Injection','Heartbleed']

    base_self.__Label_drop = []
    base_self.__Label_cnt = {}
    base_self.__Null_cnt = 0
    base_self.__FixLabel()
  #=========================================================================================================================================

  def __Print(base_self, str) -> None:
    if base_self.__PRINT_ABLE:
      print(str)
  #=========================================================================================================================================
  
  def __FixLabel(base_self):
    for x in base_self.__Label_map:
      y = base_self.__Label_map[x]
      if y not in base_self.__Real_cnt:
        print('label map ' + x)
        print('real_cnt' + y)
        base_self.__Real_cnt[y] = 0
        base_self.__Real_cnt[y] += base_self.__Real_cnt[x]
        base_self.__Real_cnt.pop(x)
    base_self.__Print("True count:")
    base_self.__Print(base_self.__Real_cnt)
  #=========================================================================================================================================

  def __ReDefineLabel_by_Category(base_self):
    base_self.__data_df.rename(columns = {" Label": "Label"}, inplace = True)
    base_self.__data_df['Category'] = base_self.__data_df['Label'].apply(lambda x: base_self.__Category_map[x] if x in base_self.__Category_map else x)
    # data_df.drop(data_df[data_df['Label'] not in __Category_map].index, inplace = True)
    return base_self.__data_df

  #=========================================================================================================================================

  def __load_raw_default(base_self, dir_path, limit_cnt:sys.maxsize, frac = None):

    base_self.__Label_cnt = {}
    base_self.__Null_cnt = 0

    tt_time = time.time()
    df_ans = pd.DataFrame()
    for root, _, files in os.walk(dir_path):
        for file in files:
          base_self.__Print("Begin file:" + file)
          if not file.endswith(".csv"):
            continue
          list_ss = []
          time_file = time.time()
          for chunk in pd.read_csv(os.path.join(root,file), index_col=None, header=0, chunksize=10000, low_memory=False):
            dfse = chunk[' Label'].value_counts()
            chunk = chunk.dropna()
            for x in dfse.index:
              sub_set = chunk.query("` Label` == @x")
              x_cnt = float(dfse[x])
              if x in base_self.__Label_map:
                x = base_self.__Label_map[x]
              if x not in base_self.__Label_cnt:
                base_self.__Label_cnt[x] = 0
              if limit_cnt == base_self.__Label_cnt[x] :
                continue

              max_cnt_chunk = min(int(limit_cnt * (x_cnt / base_self.__Real_cnt[x]) +1), sub_set.shape[0])
              
              if frac != None:
                max_cnt_chunk = min(int(frac * x_cnt + 1), sub_set.shape[0])
              
              
              max_cnt_chunk = min(max_cnt_chunk, limit_cnt - base_self.__Label_cnt[x])
              sub_set = sub_set.sample(n=max_cnt_chunk,replace = False, random_state = SEED)
              # sub_set[' Label'] = sub_set[' Label'].apply(lambda y: base_self.__Label_map[y] if y in base_self.__Label_map else y)
              list_ss.append(sub_set)
              base_self.__Label_cnt[x] += sub_set.shape[0]

          df_ans = CustomMerger().fit_transform([df_ans] + list_ss)
      
          base_self.__Print("Update label")
          base_self.__Print(base_self.__Label_cnt)

          base_self.__Print("Time load:" + str(time.time() - time_file))
          base_self.__Print(f"========================== Finish {file} =================================")

    base_self.__Print("Total time load:" + str(time.time() - tt_time))
    base_self.__data_df = df_ans
  #=========================================================================================================================================

  def __download(base_self, url, filename):
    import functools
    import pathlib
    import shutil
    import requests
    
    r = requests.get(url, stream=True, allow_redirects=True, verify = False)
    if r.status_code != 200:
      r.raise_for_status()  # Will only raise for 4xx codes, so...
      raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))
    print("Start download file - Total file_size:", file_size)
    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
      with path.open("wb") as f:
        shutil.copyfileobj(r_raw, f)

    return path  
  #=========================================================================================================================================
   
  def Down_Load_Data(base_self, datadir = os.getcwd(),  load_type="preload"):

    if load_type=="preload":
      datapath = os.path.join(datadir, "CICIDS2017_clean_data_ver_30k_42seed.csv")
      data_url = "https://s3-hcm-r1.longvan.net/19414866-ids-datasets/CICIDS2017/CICIDS2017_clean_data_ver_30k_42seed.csv"
      if os.path.exists(datapath) == True:
        print("Data already!!! No need to download.")
        return
      else:
        print("=================================== File Data not found!!! Start downloading ===================================")
        print("File saved at:", base_self.__download(data_url, datapath))
        print("================================ End download data ================================")
        return 
    
    if load_type=="raw":
      data_dir = os.path.join(datadir, "CICIDS2017")
      data_file = os.path.join(datadir, "CIC-IDS-2017.zip")
      data_url = "https://s3-hcm-r1.longvan.net/19414866-ids-datasets/CIC_IDS_2017/CIC-IDS-2017.zip"
      if os.path.exists(data_dir) == True:
        print("Data already!!! No need to download.")
        return
      else:
        print("=================================== Folder Data not found!!! Start downloading ===================================")
        if os.path.exists(data_file) == False:
          print("=================================== File Data Zip not found!!! Start downloading ===================================")
          print("File saved at:", base_self.__download(data_url, data_file))
          print("================================ End download data ================================")
        if is_zipfile(data_file):
          print("=================================== Unzipping Data!!!===================================")
          with ZipFile(data_file,"r") as zip_ref:
            for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
              zip_ref.extract(member=file)
          print("File saved at:", datadir)
          print("=================================== End download data ===================================")
        else:
          print("=================================== Zip file not valid!!! ===================================")
      return 

  #=========================================================================================================================================

  def Load_Data(base_self, datadir = os.getcwd(),  load_type="preload", limit_cnt=sys.maxsize, frac = None):

    if load_type=="preload":
      datapath = os.path.join(datadir, "CICIDS2017_clean_data_ver_30k_42seed.csv")
      if os.path.exists(datapath) == True:
        print("================================ Start load data ================================")
        base_self.__data_df =  pd.read_csv(datapath, index_col=None, header=0)
        base_self.__ReDefineLabel_by_Category()
        print("================================ Data loaded ================================")
        return
      else:
        print("=================================== File Data not found!!! ===================================")
        return 
    
    if load_type=="raw":
      datapath = os.path.join(datadir, "CICIDS2017")
      if os.path.exists(datapath) == True:
        print("================================ Start load data ================================")
        base_self.__FixLabel()
        base_self.__load_raw_default(datapath, limit_cnt, frac)
        base_self.__ReDefineLabel_by_Category()
        print("================================ Data loaded ================================")
        return
      else:
        print("=================================== Folder Data not found!!! Please download first ===================================")
      return 
  #=========================================================================================================================================
  def Show_basic_analysis(base_self):
    print("=================================== Show basic analysis ===================================")
    print("=================================== Dataframe be like:")
    print('\n' + tabulate(base_self.__data_df.head(5), headers='keys', tablefmt='psql'))
    print("=================================== Data info:")
    print(base_self.__data_df.info())
    print("=================================== Label distribution")
    print(base_self.__data_df['Label'].value_counts())
    # plt.show()
  #=========================================================================================================================================
  
  def To_dataframe(base_self):
    return base_self.__data_df