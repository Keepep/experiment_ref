
from utils import *
import pandas as pd


def main(file_name,org_path,train_path):
    method='counterfactual'
    data_name = 'MLP_keras'

    saved_path='saved_classifier/' + data_name + '_' + file_name + '.h5'
    x_train,features= tabulr_train_load(train_path)


    model = load_model(saved_path)

    # org_data=pd.read_csv(org_path)
    # org_data = org_data.to_numpy()
    file_list=os.listdir(org_path)
    file_list=np.sort(file_list)


    for file in file_list:

        org_data=pd.read_csv(org_path+file)
        org_data=org_data.to_numpy()

        pred_class=model.predict(org_data).argmax()
        pred_prob=model.predict(org_data).max()
        print(pred_prob)
        print(pred_class)


        shape = np.shape(org_data)
        target_proba = 1.0
        tol = 0.01  # want counterfactuals with p(class)>0.99
        target_class = 'other'  # any class other than 7 will do
        max_iter = 1000
        lam_init = 1e-1
        max_lam_steps = 10
        learning_rate_init = 0.1
        feature_range = (x_train.min(), x_train.max())

        cf = CounterFactual(model, shape=shape, target_proba=target_proba, tol=tol,
                            target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                            max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                            feature_range=feature_range)
        explanation =cf.explain(org_data)

        try:
            print('Pertinent negative prediction: {}'.format(explanation['cf']['class']))
        except:
            continue
        print('Original instance: {}'.format(org_data))
        print('Predicted class: {}'.format(pred_class))
        print('Pertinent negative: {}'.format(explanation['cf']['X']))
        print('Predicted class: {}'.format(explanation['cf']['class']))


        min ,ran=tabular_data_info(train_path)

        delta_data=np.abs(org_data-explanation['cf']['X'])
        delta_data=np.where( delta_data >1e-6 , delta_data,0)
        feat = np.reshape(np.array(features), (len(features)))

        if 'HELOC' in file_name :
            base_path='Results/HELOC'
        elif 'UCI' in file_name:
            base_path='Results/UCI_Credit_Card'
        base_path=base_path+method
        arg_path=(file)[:-4]
        os.system("mkdir -p {}/{}/".format(base_path,arg_path))
        save_csv(org_data,base_path,arg_path,'org',features)
        save_csv(explanation['cf']['X'], base_path, arg_path, 'per',features)
        save_csv(delta_data, base_path, arg_path, 'delta',features)


        output_org=model.predict(org_data).max()
        pred_comp_class=explanation['cf']['class']
        result = 'output_org: ' + str(output_org) + '\n' + \
                'class_org: '+str(pred_class)+'\n'+\
                 'output_comp: ' + str(explanation['cf']['proba'][0][pred_comp_class])+'\n'+ \
                 'class_comp: ' + str(explanation['cf']['class'])

        f = open(base_path+'/'+arg_path+'/' + 'result.txt', 'w')
        f.write(result)
        f.close()

if __name__ == '__main__':
    #examples/UCI_Credit_Card_test/sample0_0.csv
    #examples/HELOC_test/sample0_0.csv
    org_path='examples/UCI_Credit_Card_test/'

    #UCI_Credit_Card_IDRemoved
    #HELOC_allRemoved
    file_name='UCI_Credit_Card_IDRemoved'

    #saved_train/HELOC_allRemoved_train.csv
    #saved_train/UCI_Credit_Card_IDRemoved_train.csv
    train_path='saved_train/UCI_Credit_Card_IDRemoved_train.csv'
    main(file_name,org_path,train_path)