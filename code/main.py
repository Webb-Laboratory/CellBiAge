import Model
from dataset import *
from train import *


def main():
    # define save directory
    save_dir = "../results"

    # define dataset
    train_X, test_X, train_y, test_y, dataset_name = one_hot_binarized_dataset()
    # train_X, test_X, train_y, test_y, dataset_name = PCA_dataset(without_cate_binarized_dataset, n_components=100)    # if want PCA

    # define train parameters
    parameters = train_params(
        learning_rate=0.0001,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
        epochs=100,
        batch_size=10000
    )

    # define your model
    latent_num = 400

    # model = Model.Dense_AE(latent_num)
    # model_name = "AE_{}_with_binarized".format(latent_num)
    
    # # Start Training, after trainning, all records will be dumped to the `save_dir/model_name` dir
    # train(os.path.join(save_dir, model_name), model, parameters, train_X, train_y, test_X , test_y, dataset_name)
    


    # #ae encode
    ae_model = tf.keras.models.load_model("../results/AE_{}_with_binarized/model".format(latent_num)) 
    
    ae_train_X = ae_model.encoder(train_X)
    ae_test_X = ae_model.encoder(test_X)

    
    model = Model.Baseline_MLP(feature_nums= [200, 50, 25], dropout_rate=0.25)
    model_name = "AE_with_mlp_{}_binarized".format(latent_num)   # !!!!!! Be sure to change it every time, or your local record will be overwritten
    train(os.path.join(save_dir, model_name), model, parameters, ae_train_X, train_y, ae_test_X , test_y, dataset_name)


    
if __name__ == '__main__':
    main()
