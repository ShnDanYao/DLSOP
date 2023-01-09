import argparse
import yaml

def Config_check(config):
    #待写对参数的一些检查，比如num_class要和device id 里面的数量一致
    return 

def Config(config_dir,yaml_change):
    with open(config_dir) as f:
        config = yaml.load(f, Loader=yaml.FullLoader) 
    dictmodify=DictModify()
    config = dictmodify.Yaml_Change(config, yaml_change)
    Config_check(config)
    return config

class DictModify():
    def __init__(self):
        return 

    def typetrans(self,value):
        types,value = value.split(':')
        if types == 'f':
            value = float(value)
        elif types == 'd':
            value = int(value)
        elif types == 's':
            value = value
        elif types == 'Ls':
            if ',' in value:
                value = value.split[',']
            else:
                tmp = value
                value = []
                value.append(tmp)
        elif types == 'Ld':
            if ',' in value:
                value = value.split[',']
            else:
                tmp = value
                value = []
                value.append(tmp)
            value = [int(v) for v in value]
        elif types == 'Lf':
            if ',' in value:
                value = value.split[',']
            else:
                tmp = value
                value = []
                value.append(tmp)
            value = [float(v) for v in value]
        elif types =='b':
            if value == 'false':
                value = False
            elif value == 'true':
                value = True
        elif types == 'LLsd':
            print('error')
        return value

    def Yaml_Change(self,source,modifys):
        def cloop(uperconfig,change_split): 
            if '.' in change_split:
                key,change_revel = change_split.split('.',maxsplit=1)
                if key in uperconfig.keys():
                    uperconfig[key]=cloop(uperconfig[key], change_revel)
                    return uperconfig
                else:
                    print('key:' + key +' error')
                    exit()
            else:
                key,value = change_split.split('=')
                value = self.typetrans(value)
                if key in uperconfig.keys():
                    uperconfig[key]=value 
                else:
                    print('key:' + key +' error')
                    exit()
                return uperconfig
        for modify in modifys:
            cloop(source ,modify)	
        return source


def get_parse():
    args = argparse.ArgumentParser()

    args.add_argument("--data_argu",
            type=str,
            default='all_nor'
            )
    args.add_argument("--epoch",
        type=int,
        default=30,
        help="epoch of training")

    args.add_argument("-v", "--version",
        dest='version',
        action='store_true',
        help="show the version number of this program and exit.")

    args.add_argument("--train_dis",
            nargs='+',
            default=["2ft"]#,"8ft","14ft","20ft","26ft",
                #"32ft","38ft","44ft","50ft","56ft","62ft"]
            )
    args.add_argument("--test_dis",
            nargs='+',
            default=["2ft"]#,"8ft","14ft","20ft","26ft",
                #"32ft","38ft","44ft","50ft","56ft","62ft"]
            )
    args.add_argument("--margin",
            type=int, 
            default=128
            )
    args.add_argument("--batch_size",
            type=int, 
            default=1024
            )
    args.add_argument("--enlarge_n",
            type=int, 
            default=1
            )
    args.add_argument("--log_dir",
            type=str,
            default="log/test.log"
            )

    args.add_argument("--read_split",
            action='store_true'
            )
    args.add_argument("--save_split",
            action='store_true'
            )
    args.add_argument("--lr",
           type = float ,
           default = 0.00001
            )
    args.add_argument("--lr_step",
           type = int ,
           default = 25
            )
    args = args.parse_args()
    return args


