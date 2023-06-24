from DataLoader.utils import *
from zipfile import ZipFile, is_zipfile
from tqdm.auto import tqdm
    
SEED = 42



class ToN_IoT():
  
  def __init__(base_self, seed = 0xDEADBEEF, print_able = True, save_csv = True) -> None:
    np.random.seed(seed)
    np.set_printoptions(suppress=True)
    base_self.__PRINT_ABLE = print_able
    base_self.__SAVE_CSV = save_csv
    base_self.__data_df = pd.DataFrame()

    base_self.__Fts_names = [ 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto', 'service',
                              'duration', 'src_bytes', 'dst_bytes', 'conn_state', 'missed_bytes',
                              'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes', 'dns_query',
                              'dns_qclass', 'dns_qtype', 'dns_rcode', 'dns_AA', 'dns_RD', 'dns_RA',
                              'dns_rejected', 'ssl_version', 'ssl_cipher', 'ssl_resumed',
                              'ssl_established', 'ssl_subject', 'ssl_issuer', 'http_trans_depth',
                              'http_method', 'http_uri', 'http_version', 'http_request_body_len',
                              'http_response_body_len', 'http_status_code', 'http_user_agent',
                              'http_orig_mime_types', 'http_resp_mime_types', 'weird_name',
                              'weird_addl', 'weird_notice', 'label', 'type'
    ]

    base_self.__Real_cnt = {'normal': 796380,
                            'backdoor': 508116,
                            'ddos': 6165008,
                            'dos': 3375328,
                            'injection': 452659, 
                            'mitm': 1052,
                            'password': 1718568,
                            'ransomware': 72805, 
                            'scanning': 7140161, 
                            'xss': 2108944
    }
    base_self.__Label_map = {
                            # 'normal': 'normal',
                            # 'backdoor': 'backdoor',
                            # 'ddos': 'ddos',
                            # 'dos': 'dos',
                            # 'injection': 'injection', 
                            # 'mitm': 'mitm',
                            # 'password': 'password',
                            # 'ransomware': 'ransomware', 
                            # 'scanning': 'scanning', 
                            # 'xss': 'xss'
    }

    base_self.__Category_map = {'normal': 0,
                                'backdoor': 1,
                                'ddos': 2,
                                'dos': 3,
                                'injection': 4, 
                                'mitm': 5,
                                'password': 6,
                                'ransomware': 7, 
                                'scanning': 8, 
                                'xss': 9
    }
    base_self.__Label_true_name = ['normal',
                                'backdoor',
                                'ddos',
                                'dos',
                                'injection', 
                                'mitm',
                                'password',
                                'ransomware', 
                                'scanning', 
                                'xss'
    ]
    base_self.__Label_drop = []
    base_self.__Label_cnt = {}
    base_self.__Null_cnt = 0
    base_self.__FixLabel()
    
  def __Print(base_self, str) -> None:
    if base_self.__PRINT_ABLE:
      print(str)

  def __add_mode_features(base_self, dataset, FLAG_GENERATING: bool = False) -> pd.DataFrame:
    base_self.__Print('Drop useless features, drop lines with NaN')
    dataset = dataset.dropna()
    base_self.__Print('\n' + tabulate(dataset.head(5), headers='keys', tablefmt='psql'))

    base_self.__Print('=================================== Apply Polynomials ===================================')
    dataset = Poly_features(dataset, include=['rssi', 'rssi_mean', 'rssi_std','rssi_median'], degree=4,
                            include_bias=True)
    
    base_self.__Print('\n' + tabulate(dataset.head(5), headers='keys', tablefmt='psql'))
    base_self.__Print('=================================== Done Apply Polynomials ===================================')

    return dataset
  
  def __FixLabel(base_self):
    for x in base_self.__Label_map:
      y = base_self.__Label_map[x]
      if y not in base_self.__Real_cnt:
        base_self.__Real_cnt[y] = 0
        base_self.__Real_cnt[y] += base_self.__Real_cnt[x]
        base_self.__Real_cnt.pop(x)
    base_self.__Print("True count:")
    base_self.__Print(base_self.__Real_cnt)

  def __ReDefineLabel_by_Category(base_self):
    # base_self.__data_df.rename(columns = {"label": "Label"}, inplace = True)
    # base_self.__data_df.rename(columns = "type", inplace = True)
    # base_self.__data_df['Category'] = base_self.__data_df['Label'].apply(lambda x: base_self.__Category_map[x] if x in base_self.__Category_map else x)
    # data_df.drop(data_df[data_df['Label'] not in __Category_map].index, inplace = True)
    return base_self.__data_df



  def __load_raw_default(base_self, dir_path, limit_cnt:sys.maxsize, frac = None):

    # if self.select == "testday":
    #   selected_dir = selected_dir[1]
    # if self.select == "trainday":
    #   selected_dir = selected_dir[2]

    base_self.__Label_cnt = {}
    base_self.__Null_cnt = 0

    tt_time = time.time()
    df_ans = pd.DataFrame()
    for root, _, files in os.walk(dir_path):
        for file in files:
            base_self.__Print("Begin file " + file)
            if not file.endswith(".csv"):
                continue
            list_ss = []
            time_file = time.time()
            for chunk in pd.read_csv(os.path.join(root,file), index_col=None, names=base_self.__Fts_names, header=0, chunksize=10000, low_memory=False):
                dfse = chunk['type'].value_counts()
                chunk = chunk.dropna()
                for x in dfse.index:
                    sub_set = chunk[chunk.type == x]
                    x_cnt = float(dfse[x])
                    if x not in base_self.__Label_cnt:
                        base_self.__Label_cnt[x] = 0
                    if limit_cnt == base_self.__Label_cnt[x] :
                        continue

                    max_cnt_chunk = min(int(limit_cnt * (x_cnt / base_self.__Real_cnt[x]) +1), sub_set.shape[0])

                    if frac != None:
                        max_cnt_chunk = min(int(frac * x_cnt + 1), sub_set.shape[0])

                    max_cnt_chunk = min(max_cnt_chunk, limit_cnt - base_self.__Label_cnt[x])
                    sub_set = sub_set.sample(n=max_cnt_chunk,replace = False, random_state = SEED)
                    # sub_set['Label'] = sub_set['Label'].apply(lambda y: Label_map[y] if y in Label_map else y)
                    list_ss.append(sub_set)
                    base_self.__Label_cnt[x] += sub_set.shape[0]

            df_ans = CustomMerger().fit_transform([df_ans] + list_ss)
        
            print("Update label")
            print(base_self.__Label_cnt)

            print("Time load:", time.time() - time_file)
            print(f"========================== Finish {file} =================================")

    
    # df_ans['Label'] = df['Label'].apply(lambda x: Label_map[x] if x in Label_map else x)
    print("Total time load:", time.time() - tt_time)
    base_self.__data_df = df_ans
          

  # def __prepare_datasets(base_self, PATH_TO_DATA) -> pd.DataFrame:
    
  #   return df

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
  #=========================================================================================================================================
  
  
  def Add_more_fts(base_self):

    #Add more features
    base_self.__Print("=================================== Add more features ===================================")
    base_self.__data_df = base_self.__add_mode_features(base_self.__data_df)
    base_self.__data_df.dropna(inplace =True)
    base_self.__data_df.drop(columns=['1'],inplace =True)
    base_self.__Print("=================================== Done Add more features ===================================")
    return


  #=========================================================================================================================================
  
  
  def Down_Load_Data(base_self, datadir = os.getcwd(),  load_type="raw"):

    if load_type=="preload":
      datapath = os.path.join(datadir, "ToN_IoT_clean_data_ver_30k_42seed.csv")
      data_url = "https://s3-hcm-r1.longvan.net/19414866-ids-datasets/ToN_IoT/ToN_IoT_clean_data_ver_30k_42seed.csv"
      if os.path.exists(datapath) == True:
        print("Data already!!! No need to download.")
        return
      else:
        print("=================================== File Data not found!!! Start downloading ===================================")
        print("File saved at:", base_self.__download(data_url, datapath))
        print("================================ End download data ================================")
        return 
    
    if load_type=="raw":
      data_dir = os.path.join(datadir, "ToN_IoT")
      data_file = os.path.join(datadir, "ToN_IoT.zip")
      data_url = "https://s3-hcm-r1.longvan.net/19414866-ids-datasets/ToN_IoT/ToN_IoT.zip"
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

  def Load_Data(base_self, datadir = os.getcwd(),  load_type="raw", limit_cnt=sys.maxsize, frac = None):

    if load_type=="preload":
      datapath = os.path.join(datadir, "ToN_IoT_clean_data_ver_30k_42seed.csv")
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
      datapath = os.path.join(datadir, "ToN_IoT")
      if os.path.exists(datapath) == True:
        print("================================ Start load data ================================")
        # base_self.__FixLabel()
        base_self.__load_raw_default(datapath, limit_cnt, frac)
        base_self.__ReDefineLabel_by_Category()
        print("================================ Data loaded ================================")
        return
      else:
        print("=================================== Folder Data not found!!! Please download first ===================================")
      return 

    #=========================================================================================================================================
  
  
  # def Train_test_split(base_self, testsize=0.2):
  #   base_self.__Print("=================================== Begin Split File ===================================")
  #   df = base_self.__data_df.drop(columns=['prr'])

  #   print("=================================== Dataframe be like:")
  #   print('\n' + tabulate(base_self.__data_df.head(5), headers='keys', tablefmt='psql'))

  #   # np_data = df.to_numpy(copy=True)
  #   X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = ['target']).to_numpy(copy=True), 
  #                                                       df['target'].to_numpy(copy=True),    
  #                                                       test_size=testsize, random_state=42)


  #   # y_train = LabelEncoder().fit_transform(y_train)
  #   # y_test = LabelEncoder().fit_transform(y_test)


  #   # X_train = StandardScaler().fit_transform(X_train)

  #   # X_test = StandardScaler().fit_transform(X_test)

  #   print("Training data shape:",X_train.shape, y_train.shape)
  #   print("Testing data shape:",X_test.shape, y_test.shape)

  #   print("Label Train count:")
  #   unique= np.bincount(y_train)
  #   print(np.asarray((unique)))
  #   print("Label Test count:")
  #   unique= np.bincount(y_test)
  #   print(np.asarray((unique)))
  #   base_self.__Print("=================================== Split File End===================================")
  #   return X_train, X_test, y_train, y_test


  #=========================================================================================================================================
  
  
  def Show_basic_analysis(base_self):
    print("=================================== Show basic analysis ===================================")
    print("=================================== Dataframe be like ===================================")
    print('\n' + tabulate(base_self.__data_df.head(5), headers='keys', tablefmt='psql'))
    print("=================================== Data info ===================================")
    print(base_self.__data_df.info())
    print("=================================== Label distribution ===================================")
    print(base_self.__data_df["type"].value_counts())
    # plt.show()


  #=========================================================================================================================================
  
  
  def To_dataframe(base_self):
    return base_self.__data_df
