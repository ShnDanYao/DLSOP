for dis in 8ft 14ft 20ft 26ft 32ft 38ft 44ft 50ft 56ft 62ft 
do
python main.py --config exps/config/common.yaml \
                      --yaml_change mlflow.exp_name=s:slice \
                                    mlflow.run_name=s:$dis \
                                    data.params.train_test.dis=Ls:$dis \
done                    
