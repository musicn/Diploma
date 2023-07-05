import clustering as cls
import data
#import my_shap
import model
import cluster_explanation as ce
from sklearn.model_selection import train_test_split

D1 = 'artificial_binary_3d_1'
D2 = 'artificial_binary_2d_1'
D3 = 'artificial_binary_2d_2'
D4 = 'artificial_binary_2d_3'
D5 = 'hearth_disease_2d'
D6 = 'artificial_binary_2d_4'
D7 = 'artificial_binary_5d_1'
datasets = [D2] # D1,D2,D3,D4,D5,D6,D7
clustering_algos = [cls.HDBSCAN_C()] # cls.MDEC(),cls.KMEANS(),cls.DBSCAN_C(),cls.HDBSCAN_C()

def get_data():
    ret = []
    for ix in range(len(datasets)):
        data_class = data.Data(datasets[ix])
        data_class.construct()
        #data_class.plot()
        X = data_class.X
        y = data_class.y
        ret.append((X,y))
    return ret
    


def test_clustering_algorithms():
    data_array = get_data()
    # na vsaki mnozici podatkov izvedi vsa grucenja in jih oceni + ploti
    for ix1 in range(len(data_array)):
        print('|********************************************|')
        print('Dataset: ' + datasets[ix1])
        for ix2 in range(len(clustering_algos)):
            print('-----------')
            print('Clustering name: ' + clustering_algos[ix2].name)
            # koliko clusterjev naj bo
            for ix3 in range(11):     
                if ix3 < 2: continue
                # izvedi
                print('Cluster_num: ' + str(ix3))
                clustering_algos[ix2].cluster(data_array[ix1][0], data_array[ix1][1], ix3)
                # oceni
                silhuette_scores = clustering_algos[ix2].evaluate_silhuette_avg()
                for ix4, score in enumerate(silhuette_scores):
                    print(str(ix4) + ' - ' + 'silhuette_score: ' + str(score) + ' // %: ' + str(clustering_algos[ix2].percentage_explained[ix4]))
                dbcv_scores = clustering_algos[ix2].evaluate_dbcv()
                for ix5, score in enumerate(dbcv_scores):
                    print(str(ix5) + ' - ' + 'dbcv_score: ' + str(score) + ' // %: ' + str(clustering_algos[ix2].percentage_explained[ix5]))
                # vizualiziraj
                clustering_algos[ix2].plot()
                # reset class attributes
                clustering_algos[ix2].reset()
                # DBSCAN doesnt need cluster_num specification

def subgroup_discovery_explanation(dataset, clustering_algo, metric):
    # pridobi podatke
    data_class = data.Data(dataset)
    data_class.construct()
    #data_class.plot()
    X = data_class.X
    y = data_class.y
    
    # stevilo gruc
    maxLabels = None
    maxScore = -99999
    for ix1 in range(11):
        if ix1 < 2: continue
        
        # gruci podatke
        clustering_algo.cluster(X, y, ix1)
        #clustering_algo.plot()

        # izberi oznake najboljsega grucenja
        if metric == 'silhuette':
            silhuette_scores = clustering_algo.evaluate_silhuette_avg()
            for ix2, score in enumerate(silhuette_scores):
                if score > maxScore:
                    maxScore = score
                    maxLabels = clustering_algo.get_labels(ix2)
        if metric == 'dbcv':
            dbcv_scores = clustering_algo.evaluate_dbcv()
        
        clustering_algo.reset()

    # vsako gruco opisi s pravili (one vs all)
    rl = ce.RULES(0)
    rl.calc_rules(X, maxLabels)
    pravila = rl.get_rules()
    accuracy = rl.get_accuracy()

    # za vsak cluster najdi medoid
    med_obj = ce.MEDOID('euclidean')
    med_obj.calc_medoid(X, maxLabels)
    medoids = med_obj.get_medoid()

    # class probabilities for every cluster
    prob_obj = ce.CLASS_PROB()
    prob_obj.calc_probs(y, maxLabels)
    probs = prob_obj.get_probs()


def main1():
    #test_clustering_algorithms()
    m1 = 'silhuette'
    m2 = 'dbcv'
    subgroup_discovery_explanation(D2, cls.KMEANS(), m1)


# def main2():
#     data_class = data.Data(D3)
#     data_class.construct()
#     data_class.plot()
#     X = data_class.X
#     y = data_class.y
    
#     # split train, test set if predicting
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # do not split for explanation, if not predicting
#     xgb_class = model.XGBOOST()
#     xgb_class.fit_xgb(X, y)
#     xgb_model = xgb_class.return_model()

#     shap_class = my_shap.SHAP(xgb_model)
#     shap_values = shap_class.calc_shap_val(X)

#     # save to file force_plot.html -> doesn't display in runtime
#     #shap_class.plot_force(shap_values, X)
#     # displays in runtime
#     #shap_class.plot_summary(shap_values, X)
#     # don't know how to read it
#     #shap_class.plot_dependence(shap_values, X, 0)
    
#     # shap values as points in space
#     #shap_class.plot_clusters_2d(shap_values, y)
#     shap_class.plot_clusters_2d(shap_values, y)

#     # za plotanje ni se PCA implementiran
#     mdec = cls.MDEC()
#     mdec.cluster(shap_values, y, num_clusters=2) # brez num_clusters bo nastavitev num_cluster=len(np.unique(target_var_vec)) 
#     mdec.plot()
#     mdec.quit()

def main():
    main1()
    #main2()


if __name__ == "__main__":
    main()