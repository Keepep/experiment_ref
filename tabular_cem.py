
from utils import *
import pandas as pd


def main(file_name,org_path,train_path):

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

        mode = 'PN'  # 'PN' (pertinent negative) or 'PP' (pertinent positive)
        shape = np.shape(org_data)  # instance shape
        kappa = .2  # minimum difference needed between the prediction probability for the perturbed instance on the
        # class predicted by the original instance and the max probability on the other classes
        # in order for the first loss term to be minimized
        beta = .1  # weight of the L1 loss term
        c_init = 10.  # initial weight c of the loss term encouraging to predict a different class (PN) or
        # the same class (PP) for the perturbed instance compared to the original instance to be explained
        c_steps = 10  # nb of updates for c
        max_iterations = 1000  # nb of iterations per value of c
        # feature_range = (np.shape(org_data)[0],np.shape(org_data)[1])  # can be either a float or array of shape (1xfeatures)
        # print(np.shape(feature_range))
        clip = (-1000., 1000.)  # gradient clipping
        lr_init = 1e-2  # initial learning rate

        # initialize CEM explainer and explain instance
        cem = CEM(model, mode, shape, kappa=kappa, beta=beta,
                  max_iterations=max_iterations, c_init=c_init, c_steps=c_steps,
                  learning_rate_init=lr_init, clip=clip)
        cem.fit(x_train, no_info_type='median')
        explanation = cem.explain(org_data)

        try:
            print('Pertinent negative prediction: {}'.format(explanation[mode + '_pred']))
        except:
            continue
        print('Original instance: {}'.format(explanation['X']))
        print('Predicted class: {}'.format(explanation['X_pred']))
        print('Pertinent negative: {}'.format(explanation[mode]))
        print('Predicted class: {}'.format(explanation[mode + '_pred']))


        min ,ran=tabular_data_info(train_path)

        delta_data=np.abs(explanation['X']-explanation[mode])
        delta_data=np.where( delta_data >1e-6 , delta_data,0)
        feat = np.reshape(np.array(features), (len(features)))

        if 'HELOC' in file_name :
            base_path='Results/HELOC'
        elif 'UCI' in file_name:
            base_path='Results/UCI_Credit_Card'

        arg_path=(file)[:-4]
        os.system("mkdir -p {}/{}/".format(base_path,arg_path))
        save_csv(explanation['X'],base_path,arg_path,'org',features)
        save_csv(explanation[mode], base_path, arg_path, 'per',features)
        save_csv(delta_data, base_path, arg_path, 'delta',features)


        output_org=model.predict(org_data).max()
        output_comp=model.predict(explanation[mode]).max()

        result = 'output_org: ' + str(output_org) + '\n' + \
                'class_org: '+str(explanation['X_pred'])+'\n'+\
                 'output_comp: ' + str(output_comp)+'\n'+ \
                 'class_comp: ' + str(explanation[mode+'_pred'])

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