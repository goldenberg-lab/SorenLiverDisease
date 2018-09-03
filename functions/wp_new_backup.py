import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from fancyimpute import MICE
import numpy.core.defchararray as nd

# my libraries
from get_features import get_features
from algorithm import Alg

def run_everything(imputeMethod, Xp, targets, groups2): 
    count = 0;
    svm = Alg(); rfc = Alg(); gbc = Alg(); log = Alg(); knn = Alg(); mlp = Alg(); ens50 = Alg(); ens40=Alg(); ens30=Alg(); ens20=alg();
    svm_preds=[]; svm_f1s = []; svm_precs = []; svm_recs = []; svm_specs = []; svm_accs = []; svm_fps = []; svm_fns = []; svm_probs = []; svm_aucs = [];
    rfc_preds=[]; rfc_f1s = []; rfc_precs = []; rfc_recs = []; rfc_specs = []; rfc_accs = []; rfc_fps = []; rfc_fns = []; rfc_probs = []; rfc_aucs = [];
    gbc_preds=[]; gbc_f1s = []; gbc_precs = []; gbc_recs = []; gbc_specs = []; gbc_accs = []; gbc_fps = []; gbc_fns = []; gbc_probs = []; gbc_aucs = []; 
    log_preds=[]; log_f1s = []; log_precs = []; log_recs = []; log_specs = []; log_accs = []; log_fps = []; log_fns = []; log_probs = []; log_aucs = [];
    knn_preds=[]; knn_f1s = []; knn_precs = []; knn_recs = []; knn_specs = []; knn_accs = []; knn_fps = []; knn_fns = []; knn_probs = []; knn_aucs = [];
    mlp_preds=[]; mlp_f1s = []; mlp_precs = []; mlp_recs = []; mlp_specs = []; mlp_accs = []; mlp_fps = []; mlp_fns = []; mlp_probs = []; mlp_aucs = [];
    ens_preds=[]; ens_f1s = []; ens_precs = []; ens_recs = []; ens_specs = []; ens_accs = []; ens_fps = []; ens_fns = []; ens_probs = []; ens_aucs = []; 
    ens45_preds=[]; ens45_f1s = []; ens45_precs = []; ens45_recs = []; ens45_accs = []; ens45_fps = []; ens45_fns = []; ens45_probs = []; ens45_aucs = []; 
    ens40_preds=[]; ens40_f1s = []; ens40_precs = []; ens40_recs = []; ens40_accs = []; ens40_fps = []; ens40_fns = []; ens40_probs = []; ens40_aucs = []; 
    ens35_preds=[]; ens35_f1s = []; ens35_precs = []; ens35_recs = []; ens35_accs = []; ens35_fps = []; ens35_fns = []; ens35_probs = []; ens35_aucs = []; 
    ens30_preds=[]; ens30_f1s = []; ens30_precs = []; ens30_recs = []; ens30_accs = []; ens30_fps = []; ens30_fns = []; ens30_probs = []; ens30_aucs = []; 
    ens25_preds=[]; ens25_f1s = []; ens25_precs = []; ens25_recs = []; ens25_accs = []; ens25_fps = []; ens25_fns = []; ens25_probs = []; ens25_aucs = []; 
    ens20_preds=[]; ens20_f1s = []; ens20_precs = []; ens20_recs = []; ens20_accs = []; ens20_fps = []; ens20_fns = []; ens20_probs = []; ens20_aucs = []; 

    apri_preds=[]; apri_f1s = []; apri_precs = []; apri_recs = []; apri_specs=[]; apri_accs = []; apri_fps = []; apri_fns = []; apri_probs = []; apri_aucs = []; 
    astalt_preds=[]; astalt_f1s = []; astalt_precs = []; astalt_recs = []; astalt_specs=[]; astalt_accs = []; astalt_fps = []; astalt_fns = []; astalt_probs = []; astalt_aucs = []; 
    fib4_preds=[];   fib4_f1s = []; fib4_precs = []; fib4_recs = []; fib4_specs=[]; fib4_accs = []; fib4_fps = []; fib4_fns = []; fib4_probs = []; fib4_aucs = []; 
    Yp_tests =[]; Yp_tests_MRNDates = []; 
    
    #Xp_imputed = MICE().complete(Xp)
    Xp_imputed=Xp

    for train_index, test_index in kf.split(Xp_imputed, targets, groups2):
        print(count)
        count = count + 1;
    
        Xp_tr, Xp_ts = Xp_imputed[train_index], Xp_imputed[test_index]
        Yp_train, Yp_test = targets[train_index], targets[test_index]
        
        from sklearn.preprocessing import Imputer
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        Xp_tr = imp.fit_transform(Xp_tr)
        Xp_ts = imp.transform(Xp_ts)
        
        Yp_tests.append(Yp_test[:,0])
        Yp_tests_MRNDates.append(Yp_test[:,1])
                        
        Xp_train_unscaled = Xp_tr; 
        Xp_test_unscaled = Xp_ts;    
        
        Xp_train = Xp_tr
        Xp_test = Xp_ts
        
        #Xp_train = get_transformations(Xp_train)
        #Xp_test = get_transformations(Xp_test)
            
        # Apply Feature Scaling (But not for RFC, so that we can understand the decision tree)
        from sklearn.preprocessing import StandardScaler
        sc_p = StandardScaler()
        Xp_train = sc_p.fit_transform(Xp_train)    
        Xp_test = sc_p.transform(Xp_test)
        
        from sklearn.metrics import roc_auc_score # For calculating AUROC

        svm_pred, svm_cm, svm_proba, svm_args = svm_class(Xp_train[:,sv], Xp_test[:,sv], Yp_train[:,0].astype(int), Yp_test[:,0].astype(int))
        svm_f1,svm_reca,svm_spec,svm_prec,svm_accu,svm_fp,svm_fn = calculate_metrics(svm_cm, 'svm', None)
        svm_f1s.append(svm_f1); svm_precs.append(svm_prec); svm_recs.append(svm_reca); svm_specs.append(svm_spec); svm_accs.append(svm_accu); svm_fps.append(svm_fp); svm_fns.append(svm_fn); svm_probs.append(svm_proba);
        svm_preds.append(svm_pred)
        svm_aucs.append(roc_auc_score(Yp_test[:,0].astype(int)/4, svm_proba))
        
        generate_graph=False
        rfc_pred, rfc_cm, rfc_importances, rfc_proba, rfc_args = rfc_class(Xp_train_unscaled[:,rf], Xp_test_unscaled[:,rf], Yp_train[:,0].astype(int), Yp_test[:,0].astype(int), generate_graph)
        rfc_f1,rfc_reca,rfc_spec,rfc_prec,rfc_accu,rfc_fp,rfc_fn = calculate_metrics(rfc_cm, 'rfc', None)
        rfc_f1s.append(rfc_f1); rfc_precs.append(rfc_prec); rfc_recs.append(rfc_reca); rfc_specs.append(rfc_spec);rfc_accs.append(rfc_accu); rfc_fps.append(rfc_fp); rfc_fns.append(rfc_fn); rfc_probs.append(rfc_proba);
        rfc_preds.append(rfc_pred)
        rfc_aucs.append(roc_auc_score(Yp_test[:,0].astype(int)/4, rfc_proba))
        best_rfc_feature = rank_feature_importance("RFC", rfc_importances, rf)
                
        gbc_pred, gbc_cm, gbc_proba, gbc_args = gbc_class(Xp_train[:,gb], Xp_test[:,gb], Yp_train[:,0].astype(int), Yp_test[:,0].astype(int))
        gbc_f1,gbc_reca,gbc_spec,gbc_prec,gbc_accu,gbc_fp,gbc_fn = calculate_metrics(gbc_cm, 'gbc', None)
        gbc_f1s.append(gbc_f1); gbc_precs.append(gbc_prec); gbc_recs.append(gbc_reca); gbc_specs.append(gbc_spec); gbc_accs.append(gbc_accu); gbc_fps.append(gbc_fp); gbc_fns.append(gbc_fn); gbc_probs.append(gbc_proba);
        gbc_preds.append(gbc_pred)
        gbc_aucs.append(roc_auc_score(Yp_test[:,0].astype(int)/4, gbc_proba))

        log_pred, log_cm, log_proba, log_args = log_class(Xp_train[:,lg], Xp_test[:,lg], Yp_train[:,0].astype(int), Yp_test[:,0].astype(int))
        log_f1,log_reca,log_spec,log_prec,log_accu,log_fp,log_fn = calculate_metrics(log_cm, 'log', None)
        log_f1s.append(log_f1); log_precs.append(log_prec); log_recs.append(log_reca); log_specs.append(log_spec); log_accs.append(log_accu); log_fps.append(log_fp); log_fns.append(log_fn); log_probs.append(log_proba);
        log_preds.append(log_pred)
        log_aucs.append(roc_auc_score(Yp_test[:,0].astype(int)/4, log_proba))
        
        knn_pred, knn_cm, knn_proba, knn_args = knn_class(Xp_train[:,kn], Xp_test[:,kn], Yp_train[:,0].astype(int), Yp_test[:,0].astype(int))
        knn_f1,knn_reca,knn_spec,knn_prec,knn_accu,knn_fp,knn_fn = calculate_metrics(knn_cm, 'knn', None)
        knn_f1s.append(knn_f1); knn_precs.append(knn_prec); knn_recs.append(knn_reca);knn_specs.append(knn_spec); knn_accs.append(knn_accu); knn_fps.append(knn_fp); knn_fns.append(knn_fn); knn_probs.append(knn_proba);
        knn_preds.append(knn_pred)
        knn_aucs.append(roc_auc_score(Yp_test[:,0].astype(int)/4, knn_proba))
        
        mlp_pred, mlp_cm, mlp_proba, mlp_args = mlp_class(Xp_train[:,ml], Xp_test[:,ml], Yp_train[:,0].astype(int), Yp_test[:,0].astype(int))
        mlp_f1,mlp_reca,mlp_spec, mlp_prec,mlp_accu,mlp_fp,mlp_fn = calculate_metrics(mlp_cm, 'mlp', None)
        mlp_f1s.append(mlp_f1); mlp_precs.append(mlp_prec); mlp_recs.append(mlp_reca); mlp_specs.append(mlp_spec); mlp_accs.append(mlp_accu); mlp_fps.append(mlp_fp); mlp_fns.append(mlp_fn); mlp_probs.append(mlp_proba);
        mlp_preds.append(mlp_pred)
        mlp_aucs.append(roc_auc_score(Yp_test[:,0].astype(int)/4, mlp_proba))        
        
        #ens_pred, ens_cm = ens_class(svm_pred,rfc_pred,gbc_pred,log_pred,knn_pred, mlp_pred, Yp_test[:,0].astype(int))
        ens_pred, ens_cm, ens_proba, ens_args = ens_class(svm_proba,rfc_proba,gbc_proba,log_proba,knn_proba, mlp_proba, Yp_test[:,0].astype(int), 0.5)
        ens_f1,ens_reca,ens_spec, ens_prec,ens_accu,ens_fp,ens_fn = calculate_metrics(ens_cm, 'ens', None)
        ens_f1s.append(ens_f1); ens_precs.append(ens_prec); ens_recs.append(ens_reca); ens_specs.append(ens_spec); ens_accs.append(ens_accu); ens_fps.append(ens_fp); ens_fns.append(ens_fn);
        ens_preds.append(ens_pred)
        ens_aucs.append(roc_auc_score(Yp_test[:,0].astype(int)/4, ens_proba))

#        ens45_pred, ens45_cm, ens45_proba, ens45_args = ens_class(svm_proba,rfc_proba,gbc_proba,log_proba,knn_proba, mlp_proba, Yp_test[:,0].astype(int), 0.45)
#        ens45_f1,ens45_reca,ens45_prec,ens45_accu,ens45_fp,ens45_fn = calculate_metrics(ens45_cm, 'ens45', None)
#        ens45_f1s.append(ens45_f1); ens45_precs.append(ens45_prec); ens45_recs.append(ens45_reca); ens45_accs.append(ens45_accu); ens45_fps.append(ens45_fp); ens45_fns.append(ens45_fn);
#        ens45_preds.append(ens45_pred)
#        ens45_aucs.append(roc_auc_score(Yp_test[:,0].astype(int)/4, ens45_proba))
#        
#        ens40_pred, ens40_cm, ens40_proba, ens40_args = ens_class(svm_proba,rfc_proba,gbc_proba,log_proba,knn_proba, mlp_proba, Yp_test[:,0].astype(int), 0.4)
#        ens40_f1,ens40_reca,ens40_prec,ens40_accu,ens40_fp,ens40_fn = calculate_metrics(ens40_cm, 'ens40', None)
#        ens40_f1s.append(ens40_f1); ens40_precs.append(ens40_prec); ens40_recs.append(ens40_reca); ens40_accs.append(ens40_accu); ens40_fps.append(ens40_fp); ens40_fns.append(ens40_fn);
#        ens40_preds.append(ens40_pred)
#        ens40_aucs.append(roc_auc_score(Yp_test[:,0].astype(int)/4, ens40_proba))
#        
#        ens35_pred, ens35_cm, ens35_proba, ens35_args = ens_class(svm_proba,rfc_proba,gbc_proba,log_proba,knn_proba, mlp_proba, Yp_test[:,0].astype(int), 0.35)
#        ens35_f1,ens35_reca,ens35_prec,ens35_accu,ens35_fp,ens35_fn = calculate_metrics(ens35_cm, 'ens35', None)
#        ens35_f1s.append(ens35_f1); ens35_precs.append(ens35_prec); ens35_recs.append(ens35_reca); ens35_accs.append(ens35_accu); ens35_fps.append(ens35_fp); ens35_fns.append(ens35_fn);
#        ens35_preds.append(ens35_pred)
#        ens35_aucs.append(roc_auc_score(Yp_test[:,0].astype(int)/4, ens35_proba))
#        
#        ens30_pred, ens30_cm, ens30_proba, ens30_args = ens_class(svm_proba,rfc_proba,gbc_proba,log_proba,knn_proba, mlp_proba, Yp_test[:,0].astype(int), 0.3)
#        ens30_f1,ens30_reca,ens30_prec,ens30_accu,ens30_fp,ens30_fn = calculate_metrics(ens30_cm, 'ens30', None)
#        ens30_f1s.append(ens30_f1); ens30_precs.append(ens30_prec); ens30_recs.append(ens30_reca); ens30_accs.append(ens30_accu); ens30_fps.append(ens30_fp); ens30_fns.append(ens30_fn);
#        ens30_preds.append(ens30_pred)
#        ens30_aucs.append(roc_auc_score(Yp_test[:,0].astype(int)/4, ens30_proba))
#        
#        ens25_pred, ens25_cm, ens25_proba, ens25_args = ens_class(svm_proba,rfc_proba,gbc_proba,log_proba,knn_proba, mlp_proba, Yp_test[:,0].astype(int), 0.25)
#        ens25_f1,ens25_reca,ens25_prec,ens25_accu,ens25_fp,ens25_fn = calculate_metrics(ens25_cm, 'ens25', None)
#        ens25_f1s.append(ens25_f1); ens25_precs.append(ens25_prec); ens25_recs.append(ens25_reca); ens25_accs.append(ens25_accu); ens25_fps.append(ens25_fp); ens25_fns.append(ens25_fn);
#        ens25_preds.append(ens25_pred)
#        ens25_aucs.append(roc_auc_score(Yp_test[:,0].astype(int)/4, ens25_proba))
#        
#        ens20_pred, ens20_cm, ens20_proba, ens20_args = ens_class(svm_proba,rfc_proba,gbc_proba,log_proba,knn_proba, mlp_proba, Yp_test[:,0].astype(int), 0.2)
#        ens20_f1,ens20_reca,ens20_prec,ens20_accu,ens20_fp,ens20_fn = calculate_metrics(ens20_cm, 'ens20', None)
#        ens20_f1s.append(ens20_f1); ens20_precs.append(ens20_prec); ens20_recs.append(ens20_reca); ens20_accs.append(ens20_accu); ens20_fps.append(ens20_fp); ens20_fns.append(ens20_fn);
#        ens20_preds.append(ens20_pred)
#        ens20_aucs.append(roc_auc_score(Yp_test[:,0].astype(int)/4, ens20_proba))
        
        apri_pred, apri_cm = apri_class(Xp_test_unscaled[:,apr], Yp_test[:,0].astype(int));
        apri_f1,apri_reca, apri_spec, apri_prec,apri_accu,apri_fp,apri_fn = calculate_metrics(apri_cm, 'apri', None)
        apri_f1s.append(apri_f1); apri_precs.append(apri_prec); apri_recs.append(apri_reca);apri_specs.append(apri_spec); apri_accs.append(apri_accu); apri_fps.append(apri_fp); apri_fns.append(apri_fn); 
        apri_preds.append(apri_pred)
        
        astalt_pred, astalt_cm = astalt_class(Xp_test_unscaled[:,asal], Yp_test[:,0].astype(int))
        astalt_f1,astalt_reca,astalt_spec, astalt_prec,astalt_accu,astalt_fp,astalt_fn = calculate_metrics(astalt_cm, 'astalt', None)
        astalt_f1s.append(astalt_f1); astalt_precs.append(astalt_prec); astalt_recs.append(astalt_reca); astalt_specs.append(astalt_spec); astalt_accs.append(astalt_accu); astalt_fps.append(astalt_fp); astalt_fns.append(astalt_fn); 
        astalt_preds.append(astalt_pred)

        fib4_pred, fib4_cm = fib4_class(Xp_test_unscaled[:,fb4], Yp_test[:,0].astype(int))
        fib4_f1,fib4_reca, fib4_spec, fib4_prec,fib4_accu,fib4_fp,fib4_fn = calculate_metrics(fib4_cm, 'fib4', None)
        fib4_f1s.append(fib4_f1); fib4_precs.append(fib4_prec); fib4_recs.append(fib4_reca); fib4_specs.append(fib4_spec); fib4_accs.append(fib4_accu); fib4_fps.append(fib4_fp); fib4_fns.append(fib4_fn); 
        fib4_preds.append(fib4_pred)

    print_results("SVM", svm_args, sv, svm_f1s, svm_recs, svm_specs, svm_precs, svm_accs, svm_fns, svm_fps, svm_aucs)
    print_results("RFC", rfc_args, rf, rfc_f1s, rfc_recs, rfc_specs, rfc_precs, rfc_accs, rfc_fns, rfc_fps, rfc_aucs)
    print_results("GBC", gbc_args, gb, gbc_f1s, gbc_recs, gbc_specs, gbc_precs, gbc_accs, gbc_fns, gbc_fps, gbc_aucs)
    print_results("LOG", log_args, lg, log_f1s, log_recs, log_specs, log_precs, log_accs, log_fns, log_fps, log_aucs)
    print_results("KNN", knn_args, kn, knn_f1s, knn_recs, knn_specs, knn_precs, knn_accs, knn_fns, knn_fps, knn_aucs)
    print_results("MLP", mlp_args, ml, mlp_f1s, mlp_recs, mlp_specs, mlp_precs, mlp_accs, mlp_fns, mlp_fps, mlp_aucs)
    print_results("ENS", ens_args, None, ens_f1s, ens_recs, ens_specs, ens_precs, ens_accs, ens_fns, ens_fps, ens_aucs)     
    print_results("APRI", None, apr, apri_f1s, apri_recs, apri_specs, apri_precs, apri_accs, apri_fns, apri_fps, apri_aucs)
    print_results("ASTALT", None, asal, astalt_f1s, astalt_recs, astalt_specs, astalt_precs, astalt_accs, astalt_fns, astalt_fps, astalt_aucs)
    print_results("FIB4", None, fb4, fib4_f1s, fib4_recs, fib4_specs, fib4_precs, fib4_accs, fib4_fns, fib4_fps, fib4_aucs)

    from beautifultable import BeautifulTable
    table = BeautifulTable(max_width=300)
    table.column_headers = [" ", "ENS(20)", "ENS(25)", "ENS(30)","ENS(35)","ENS(40)","ENS(45)","ENS(50)", "AST/ALT", "FIB4", "APRI"]
    table.append_row(['F1' ,('%.2f' % (np.mean(ens20_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(ens20_f1s)*100)),('%.2f' % (np.mean(ens25_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(ens25_f1s)*100)),('%.2f' % (np.mean(ens30_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(ens30_f1s)*100)),('%.2f' % (np.mean(ens35_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(ens35_f1s)*100)),('%.2f' % (np.mean(ens40_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(ens40_f1s)*100)),('%.2f' % (np.mean(ens45_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(ens45_f1s)*100)),('%.2f' % (np.mean(ens_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_f1s)*100)), ('%.2f' % (np.mean(astalt_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_f1s)*100)),('%.2f' % (np.mean(fib4_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_f1s)*100)),('%.2f' % (np.mean(apri_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_f1s)*100))])
    table.append_row(['Recall',('%.2f' % (np.mean(ens20_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens20_recs)*100)),('%.2f' % (np.mean(ens25_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens25_recs)*100)),('%.2f' % (np.mean(ens30_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens30_recs)*100)),('%.2f' % (np.mean(ens35_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens35_recs)*100)),('%.2f' % (np.mean(ens40_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens40_recs)*100)),('%.2f' % (np.mean(ens45_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens45_recs)*100)),('%.2f' % (np.mean(ens_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_recs)*100)),('%.2f' % (np.mean(astalt_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_recs)*100)),('%.2f' % (np.mean(fib4_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_recs)*100)), ('%.2f' % (np.mean(apri_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_recs)*100))])
    table.append_row(['Precision',('%.2f' % (np.mean(ens20_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens20_precs)*100)),('%.2f' % (np.mean(ens25_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens25_precs)*100)),('%.2f' % (np.mean(ens30_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens30_precs)*100)),('%.2f' % (np.mean(ens35_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens35_precs)*100)),('%.2f' % (np.mean(ens40_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens40_precs)*100)),('%.2f' % (np.mean(ens45_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens45_precs)*100)),('%.2f' % (np.mean(ens_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_precs)*100)),('%.2f' % (np.mean(astalt_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_precs)*100)),('%.2f' % (np.mean(fib4_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_precs)*100)),('%.2f' % (np.mean(apri_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_precs)*100))])
    table.append_row(['Accuracy',('%.2f' % (np.mean(ens20_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens20_accs)*100)),('%.2f' % (np.mean(ens25_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens25_accs)*100)),('%.2f' % (np.mean(ens30_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens30_accs)*100)),('%.2f' % (np.mean(ens35_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens35_accs)*100)),('%.2f' % (np.mean(ens40_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens40_accs)*100)),('%.2f' % (np.mean(ens45_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens45_accs)*100)),('%.2f' % (np.mean(ens_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_accs)*100)),('%.2f' % (np.mean(astalt_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_accs)*100)),('%.2f' % (np.mean(fib4_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_accs)*100)), ('%.2f' % (np.mean(apri_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_accs)*100))])
    table.append_row(['False Neg Rate',('%.2f' % (np.mean(ens20_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(ens20_fns)*100)),('%.2f' % (np.mean(ens25_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(ens25_fns)*100)),('%.2f' % (np.mean(ens30_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(ens30_fns)*100)),('%.2f' % (np.mean(ens35_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(ens35_fns)*100)),('%.2f' % (np.mean(ens40_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(ens40_fns)*100)), ('%.2f' % (np.mean(ens45_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(ens45_fns)*100)),('%.2f' % (np.mean(ens_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_fns)*100)),('%.2f' % (np.mean(astalt_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_fns)*100)),('%.2f' % (np.mean(fib4_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_fns)*100)), ('%.2f' % (np.mean(apri_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_fns)*100))])
    table.append_row(['False Pos Rate',('%.2f' % (np.mean(ens20_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(ens20_fps)*100)),('%.2f' % (np.mean(ens25_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(ens25_fps)*100)),('%.2f' % (np.mean(ens30_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(ens30_fps)*100)),('%.2f' % (np.mean(ens35_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(ens35_fps)*100)),('%.2f' % (np.mean(ens40_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(ens40_fps)*100)), ('%.2f' % (np.mean(ens45_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(ens45_fps)*100)) ,('%.2f' % (np.mean(ens_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_fps)*100)),('%.2f' % (np.mean(astalt_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_fps)*100)),('%.2f' % (np.mean(fib4_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_fps)*100)), ('%.2f' % (np.mean(apri_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_fps)*100))])
    table.append_row(['AUROC',('%.2f' % (np.mean(ens20_aucs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens20_aucs)*100)),('%.2f' % (np.mean(ens25_aucs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens25_aucs)*100)),('%.2f' % (np.mean(ens30_aucs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens30_aucs)*100)),('%.2f' % (np.mean(ens35_aucs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens35_aucs)*100)),('%.2f' % (np.mean(ens40_aucs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens40_aucs)*100)),('%.2f' % (np.mean(ens45_aucs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens45_aucs)*100)),('%.2f' % (np.mean(ens_aucs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_aucs)*100)),("NaN +/- NaN"),("NaN +/- NaN"), ("NaN +/- NaN")])
    
    f1s = [np.mean(svm_f1s), np.mean(rfc_f1s), np.mean(gbc_f1s), np.mean(log_f1s), np.mean(knn_f1s), np.mean(mlp_f1s), np.mean(ens_f1s), np.mean(astalt_f1s), np.mean(fib4_f1s), np.mean(apri_f1s)]
    accs = [np.mean(svm_accs), np.mean(rfc_accs), np.mean(gbc_accs), np.mean(log_accs), np.mean(knn_accs), np.mean(mlp_accs), np.mean(ens_accs), np.mean(astalt_accs), np.mean(fib4_accs), np.mean(apri_accs)]
    recs = [np.mean(svm_recs), np.mean(rfc_recs), np.mean(gbc_recs), np.mean(log_recs), np.mean(knn_recs), np.mean(mlp_recs), np.mean(ens_recs), np.mean(astalt_recs), np.mean(fib4_recs), np.mean(apri_recs)]
    specs = [np.mean(svm_specs), np.mean(rfc_specs), np.mean(gbc_specs), np.mean(log_specs), np.mean(knn_specs), np.mean(mlp_specs), np.mean(ens_specs), np.mean(astalt_specs), np.mean(fib4_specs), np.mean(apri_specs)]    
    precs = [np.mean(svm_precs), np.mean(rfc_precs), np.mean(gbc_precs), np.mean(log_precs), np.mean(knn_precs), np.mean(mlp_precs), np.mean(ens_precs), np.mean(astalt_precs), np.mean(fib4_precs), np.mean(apri_precs)]
    fns = [np.mean(svm_fns), np.mean(rfc_fns), np.mean(gbc_fns), np.mean(log_fns), np.mean(knn_fns), np.mean(mlp_fns), np.mean(ens_fns), np.mean(astalt_fns), np.mean(fib4_fns), np.mean(apri_fns)]
    fps = [np.mean(svm_fps), np.mean(rfc_fps), np.mean(gbc_fps), np.mean(log_fps), np.mean(knn_fps), np.mean(mlp_fps), np.mean(ens_fps), np.mean(astalt_fps), np.mean(fib4_fps), np.mean(fib4_fps)]
    aucs = [np.mean(svm_aucs), np.mean(rfc_aucs), np.mean(gbc_aucs), np.mean(log_aucs), np.mean(knn_aucs), np.mean(mlp_aucs), np.mean(ens_aucs)] # , np.mean(astalt_aucs), np.mean(fib4_aucs)

    metrics = [f1s, accs, recs, specs, precs, fns, fps, aucs]
        
    f1s_err = np.array([[np.std(svm_f1s), np.std(rfc_f1s), np.std(gbc_f1s), np.std(log_f1s), np.std(knn_f1s), np.std(mlp_f1s), np.std(ens_f1s), np.std(astalt_f1s), np.std(fib4_f1s), np.std(apri_f1s)],[np.std(svm_f1s), np.std(rfc_f1s), np.std(gbc_f1s), np.std(log_f1s), np.std(knn_f1s), np.std(mlp_f1s), np.std(ens_f1s), np.std(astalt_f1s), np.std(fib4_f1s), np.std(apri_f1s)]])
    accs_err = np.array([[np.std(svm_accs), np.std(rfc_accs), np.std(gbc_accs), np.std(log_accs), np.std(knn_accs), np.std(mlp_accs), np.std(ens_accs), np.std(astalt_accs), np.std(fib4_accs), np.std(apri_accs)],[np.std(svm_accs), np.std(rfc_accs), np.std(gbc_accs), np.std(log_accs), np.std(knn_accs), np.std(mlp_accs), np.std(ens_accs), np.std(astalt_accs), np.std(fib4_accs), np.std(apri_accs)]])
    recs_err = np.array([[np.std(svm_recs), np.std(rfc_recs), np.std(gbc_recs), np.std(log_recs), np.std(knn_recs), np.std(mlp_recs), np.std(ens_recs), np.std(astalt_recs), np.std(fib4_recs), np.std(apri_recs)],[np.std(svm_recs), np.std(rfc_recs), np.std(gbc_recs), np.std(log_recs), np.std(knn_recs), np.std(mlp_recs), np.std(ens_recs), np.std(astalt_recs), np.std(fib4_recs), np.std(apri_recs)]])
    specs_err = np.array([[np.std(svm_specs), np.std(rfc_specs), np.std(gbc_specs), np.std(log_specs), np.std(knn_specs), np.std(mlp_specs), np.std(ens_specs), np.std(astalt_specs), np.std(fib4_specs), np.std(apri_specs)],[np.std(svm_specs), np.std(rfc_specs), np.std(gbc_specs), np.std(log_specs), np.std(knn_specs), np.std(mlp_specs), np.std(ens_specs), np.std(astalt_specs), np.std(fib4_specs), np.std(apri_specs)]])
    precs_err = np.array([[np.std(svm_precs), np.std(rfc_precs), np.std(gbc_precs), np.std(log_precs), np.std(knn_precs), np.std(mlp_precs), np.std(ens_precs), np.std(astalt_precs), np.std(fib4_precs), np.std(apri_precs)],[np.std(svm_precs), np.std(rfc_precs), np.std(gbc_precs), np.std(log_precs), np.std(knn_precs), np.std(mlp_precs), np.std(ens_precs), np.std(astalt_precs), np.std(fib4_precs), np.std(apri_precs)]])
    fns_err = np.array([[np.std(svm_fns), np.std(rfc_fns), np.std(gbc_fns), np.std(log_fns), np.std(knn_fns), np.std(mlp_fns), np.std(ens_fns), np.std(astalt_fns), np.std(fib4_fns), np.std(apri_fns)],[np.std(svm_fns), np.std(rfc_fns), np.std(gbc_fns), np.std(log_fns), np.std(knn_fns), np.std(mlp_fns), np.std(ens_fns), np.std(astalt_fns), np.std(fib4_fns), np.std(apri_fns)]])
    fps_err = np.array([[np.std(svm_fps), np.std(rfc_fps), np.std(gbc_fps), np.std(log_fps), np.std(knn_fps), np.std(mlp_fps), np.std(ens_fps), np.std(astalt_fps), np.std(fib4_fps), np.std(apri_fps)],[np.std(svm_fps), np.std(rfc_fps), np.std(gbc_fps), np.std(log_fps), np.std(knn_fps), np.std(mlp_fps), np.std(ens_fps), np.std(astalt_fps), np.std(fib4_fps), np.std(apri_fps)]])
    aucs_err = np.array([[np.std(svm_aucs), np.std(rfc_aucs), np.std(gbc_aucs), np.std(log_aucs), np.std(knn_aucs), np.std(mlp_aucs), np.std(ens_aucs)],[np.std(svm_aucs), np.std(rfc_aucs), np.std(gbc_aucs), np.std(log_aucs), np.std(knn_aucs), np.std(mlp_aucs), np.std(ens_aucs)]]) # , np.std(astalt_aucs), np.std(fib4_aucs)
    
    errors = [f1s_err, accs_err, recs_err, specs_err, precs_err, fns_err, fps_err, aucs_err]
    
    print(table)
    print(svm_args)
    print(sv, sep=',')
    print(rfc_args)
    print(rf, sep=',')
    print(gbc_args)
    print(gb, sep=',')
    print(log_args)
    print(lg, sep=',')
    print(knn_args)
    print(kn, sep=',')
    print(mlp_args)
    print(ml, sep=',')
    
    from beautifultable import BeautifulTable
    table = BeautifulTable(max_width=300)
    table.column_headers = [" ", "SVM", "RFC", "GBC","LOG","KNN","MLP","ENS", "AST/ALT", "FIB4", "APRI"]
    table.append_row(['F1' ,('%.2f' % (np.mean(svm_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(svm_f1s)*100)),('%.2f' % (np.mean(rfc_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(rfc_f1s)*100)),('%.2f' % (np.mean(gbc_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(gbc_f1s)*100)),('%.2f' % (np.mean(log_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(log_f1s)*100)),('%.2f' % (np.mean(knn_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(knn_f1s)*100)),('%.2f' % (np.mean(mlp_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(mlp_f1s)*100)),('%.2f' % (np.mean(ens_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_f1s)*100)), ('%.2f' % (np.mean(astalt_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_f1s)*100)),('%.2f' % (np.mean(fib4_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_f1s)*100)),('%.2f' % (np.mean(apri_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_f1s)*100))])
    table.append_row(['Sensitivity',('%.2f' % (np.mean(svm_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(svm_recs)*100)),('%.2f' % (np.mean(rfc_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(rfc_recs)*100)),('%.2f' % (np.mean(gbc_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(gbc_recs)*100)),('%.2f' % (np.mean(log_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(log_recs)*100)),('%.2f' % (np.mean(knn_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(knn_recs)*100)),('%.2f' % (np.mean(mlp_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(mlp_recs)*100)),('%.2f' % (np.mean(ens_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_recs)*100)),('%.2f' % (np.mean(astalt_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_recs)*100)),('%.2f' % (np.mean(fib4_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_recs)*100)), ('%.2f' % (np.mean(apri_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_recs)*100))])
    table.append_row(['Specificty',('%.2f' % (np.mean(svm_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(svm_specs)*100)),('%.2f' % (np.mean(rfc_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(rfc_specs)*100)),('%.2f' % (np.mean(gbc_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(gbc_specs)*100)),('%.2f' % (np.mean(log_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(log_specs)*100)),('%.2f' % (np.mean(knn_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(knn_specs)*100)),('%.2f' % (np.mean(mlp_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(mlp_specs)*100)),('%.2f' % (np.mean(ens_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_specs)*100)),('%.2f' % (np.mean(astalt_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_specs)*100)),('%.2f' % (np.mean(fib4_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_specs)*100)), ('%.2f' % (np.mean(apri_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_specs)*100))])    
    table.append_row(['Precision',('%.2f' % (np.mean(svm_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(svm_precs)*100)),('%.2f' % (np.mean(rfc_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(rfc_precs)*100)),('%.2f' % (np.mean(gbc_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(gbc_precs)*100)),('%.2f' % (np.mean(log_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(log_precs)*100)),('%.2f' % (np.mean(knn_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(knn_precs)*100)),('%.2f' % (np.mean(mlp_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(mlp_precs)*100)),('%.2f' % (np.mean(ens_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_precs)*100)),('%.2f' % (np.mean(astalt_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_precs)*100)),('%.2f' % (np.mean(fib4_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_precs)*100)),('%.2f' % (np.mean(apri_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_precs)*100))])
    table.append_row(['Accuracy',('%.2f' % (np.mean(svm_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(svm_accs)*100)),('%.2f' % (np.mean(rfc_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(rfc_accs)*100)),('%.2f' % (np.mean(gbc_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(gbc_accs)*100)),('%.2f' % (np.mean(log_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(log_accs)*100)),('%.2f' % (np.mean(knn_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(knn_accs)*100)),('%.2f' % (np.mean(mlp_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(mlp_accs)*100)),('%.2f' % (np.mean(ens_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_accs)*100)),('%.2f' % (np.mean(astalt_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_accs)*100)),('%.2f' % (np.mean(fib4_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_accs)*100)), ('%.2f' % (np.mean(apri_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_accs)*100))])
    table.append_row(['False Neg Rate',('%.2f' % (np.mean(svm_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(svm_fns)*100)),('%.2f' % (np.mean(rfc_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(rfc_fns)*100)),('%.2f' % (np.mean(gbc_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(gbc_fns)*100)),('%.2f' % (np.mean(log_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(log_fns)*100)),('%.2f' % (np.mean(knn_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(knn_fns)*100)), ('%.2f' % (np.mean(mlp_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(mlp_fns)*100)),('%.2f' % (np.mean(ens_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_fns)*100)),('%.2f' % (np.mean(astalt_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_fns)*100)),('%.2f' % (np.mean(fib4_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_fns)*100)), ('%.2f' % (np.mean(apri_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_fns)*100))])
    table.append_row(['False Pos Rate',('%.2f' % (np.mean(svm_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(svm_fps)*100)),('%.2f' % (np.mean(rfc_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(rfc_fps)*100)),('%.2f' % (np.mean(gbc_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(gbc_fps)*100)),('%.2f' % (np.mean(log_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(log_fps)*100)),('%.2f' % (np.mean(knn_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(knn_fps)*100)), ('%.2f' % (np.mean(mlp_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(mlp_fps)*100)) ,('%.2f' % (np.mean(ens_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_fps)*100)),('%.2f' % (np.mean(astalt_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_fps)*100)),('%.2f' % (np.mean(fib4_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_fps)*100)), ('%.2f' % (np.mean(apri_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_fps)*100))])
    table.append_row(['AUROC',('%.2f' % (np.mean(svm_aucs))) + ' +/- ' + ('%.2f' % (np.std(svm_aucs))),('%.2f' % (np.mean(rfc_aucs))) + ' +/- ' + ('%.2f' % (np.std(rfc_aucs))),('%.2f' % (np.mean(gbc_aucs))) + ' +/- ' + ('%.2f' % (np.std(gbc_aucs))),('%.2f' % (np.mean(log_aucs))) + ' +/- ' + ('%.2f' % (np.std(log_aucs))),('%.2f' % (np.mean(knn_aucs))) + ' +/- ' + ('%.2f' % (np.std(knn_aucs))),('%.2f' % (np.mean(mlp_aucs))) + ' +/- ' + ('%.2f' % (np.std(mlp_aucs))),('%.2f' % (np.mean(ens_aucs))) + ' +/- ' + ('%.2f' % (np.std(ens_aucs))),("NaN +/- NaN"),("NaN +/- NaN"), ("NaN +/- NaN")])
    print(table)
        
    best_rfc_feature = rank_feature_importance("RFC", rfc_importances, rf)
    svm_preds = [item for sublist in svm_preds for item in sublist];
    rfc_preds = [item for sublist in rfc_preds for item in sublist];
    gbc_preds = [item for sublist in gbc_preds for item in sublist];
    log_preds = [item for sublist in log_preds for item in sublist];
    knn_preds = [item for sublist in knn_preds for item in sublist];
    mlp_preds = [item for sublist in mlp_preds for item in sublist];
    ens_preds = [item for sublist in ens_preds for item in sublist];
    apri_preds = [item for sublist in apri_preds for item in sublist];
    astalt_preds = [item for sublist in astalt_preds for item in sublist];
    fib4_preds = [item for sublist in fib4_preds for item in sublist];
    
    Yp_tests = [item for sublist in Yp_tests for item in sublist];
    Yp_tests_MRNDates = [item for sublist in Yp_tests_MRNDates for item in sublist];
    
    return svm_preds, rfc_preds, gbc_preds, log_preds, knn_preds, mlp_preds, ens_preds, astalt_preds, fib4_preds, apri_preds, Yp_tests, Yp_tests_MRNDates, metrics, errors


def svm_class(Xp_train, Xp_test, Yp_train, Yp_test):
    from sklearn.svm import SVC
    svm = SVC(verbose=False, gamma = 0.0025, kernel='rbf', probability=True)
    svmArgs = "SVC(verbose=False, C=10, kernel='rbf', gamma=0.0845, probability=True)"
    svm.fit(Xp_train, Yp_train)
    Yp_pred = svm.predict(Xp_test)
    probabilities = svm.predict_proba(Xp_test)
    probabilities = probabilities[:,1]
    cm = my_confusion_matrix(Yp_test, Yp_pred)    
    return Yp_pred, cm, probabilities, svmArgs

# Sens:  85.12
# ACC: 70.60
# AUROC: 78.84

def rfc_class(Xp_train, Xp_test, Yp_train, Yp_test, generate_graph):
    from sklearn.ensemble import RandomForestClassifier

    # Max_depth = 25;
    rfc = RandomForestClassifier(n_jobs=-1, criterion='gini', bootstrap=True, n_estimators=1000, random_state=0)
    #rfc = RandomForestClassifier(n_jobs=-1,max_depth=1, bootstrap=False, max_features='auto', criterion='entropy' , n_estimators=500, random_state=0)
    rfcArgs = "RandomForestClassifier(n_jobs=-1,max_depth=25, bootstrap=False, max_features='auto', criterion='entropy' , n_estimators=500, random_state=0)"
    rfc.fit(Xp_train, Yp_train)
    
    Yp_pred = rfc.predict(Xp_test)
    probabilities = rfc.predict_proba(Xp_test)
    probabilities = probabilities[:,1] # to generate AUROC, we only need positive probabilities
    
    importances = rfc.feature_importances_

    if (generate_graph == True):
        visualize_data(rfc.estimators_[-1])

    cm = my_confusion_matrix(Yp_test, Yp_pred)
    return Yp_pred, cm, importances, probabilities, rfcArgs

#66.99

def gbc_class(Xp_train, Xp_test, Yp_train, Yp_test):
    from sklearn.ensemble import GradientBoostingClassifier
    
    gbc = GradientBoostingClassifier(n_estimators=1000, random_state=0, criterion='friedman_mse', loss='exponential')
    gbcArgs = "GradientBoostingClassifier(learning_rate=1, max_depth=4, n_estimators=1000, random_state=0, criterion='friedman_mse', loss='exponential')"
    gbc.fit(Xp_train, Yp_train)
    Yp_pred = gbc.predict(Xp_test)
    probabilities = gbc.predict_proba(Xp_test)
    probabilities = probabilities[:,1] # to generate AUROC, we only need positive probabilities
    
    cm = my_confusion_matrix(Yp_test, Yp_pred)
    return Yp_pred, cm, probabilities, gbcArgs

def log_class(Xp_train, Xp_test, Yp_train, Yp_test):
    from sklearn.linear_model import LogisticRegression
    
    log = LogisticRegression(random_state = 0, max_iter =1000, tol=1e-6, C=0.5, solver='liblinear')
    logArgs = "LogisticRegression(random_state = 0, max_iter =1000, tol=1e-6, C=0.01, solver='liblinear')"
    Yp_train = Yp_train/4    
    log.fit(Xp_train, Yp_train)
    Yp_pred = log.predict(Xp_test)*4        
    probabilities = log.predict_proba(Xp_test)
    probabilities = probabilities[:,1] # to generate AUROC, we only need positive probabilities
    
    cm = my_confusion_matrix(Yp_test, Yp_pred)
    return Yp_pred, cm, probabilities, logArgs

def knn_class(Xp_train, Xp_test, Yp_train, Yp_test):
    from sklearn.neighbors import KNeighborsClassifier
    KNN = KNeighborsClassifier(n_neighbors=2, weights='distance', p=1.1)
    KNNArgs = "KNeighborsClassifier(n_neighbors=2, weights='distance', p=1.1)"
    KNN.fit(Xp_train, Yp_train)     
    Yp_pred = KNN.predict(Xp_test)
    probabilities = KNN.predict_proba(Xp_test)
    probabilities = probabilities[:,1]
        
    cm = my_confusion_matrix(Yp_test, Yp_pred)
    return Yp_pred, cm, probabilities, KNNArgs

def mlp_class(Xp_train, Xp_test, Yp_train, Yp_test):
    from sklearn.neural_network import MLPClassifier
    MLP = MLPClassifier(activation='relu', solver='adam', tol=0.0001, hidden_layer_sizes = (10), max_iter=800,learning_rate='constant', alpha=0.0005, random_state = 0)
    MLPArgs = "MLPClassifier(activation='relu', solver='adam', tol=0.0001, hidden_layer_sizes = (18), max_iter=800,learning_rate='constant', alpha=0.0005, random_state = 0)"
    MLP.fit(Xp_train, Yp_train)
    #hidden_layer_sizes=(28,16,12),
    Yp_pred = MLP.predict(Xp_test)
    probabilities = MLP.predict_proba(Xp_test)
    probabilities = probabilities[:,1]
    
    cm = my_confusion_matrix(Yp_test, Yp_pred)
    return Yp_pred, cm, probabilities, MLPArgs


def ens_class (s, r, g, l, k, a, Yp_test, thresh):
    ens = s; # Get right data type and size
    ens_proba = []
    for i in range(0,np.size(s)):
        ens_proba.append((s[i] + r[i] + l[i]+ g[i] +k[i]+ a[i])/6) 
        if (ens_proba[i] >= thresh):
            ens[i] = 4;
        else:
            ens[i] = 0;
    ensArgs = "ens_proba[i] = (s[i] + r[i] + l[i] + g[i] + k[i] + a[i])/6), (ens_proba[i] >= 0.7)"
    cm = my_confusion_matrix(Yp_test,ens)
    return ens, cm, ens_proba, ensArgs

def apri_class (Xp_test, Yp_test):
    #Yp_pred = np.zeros([len(Yp_test),1]); # initialize to right size
    Yp_pred_new = [];
    Yp_test_new = [];
    
    for i in range(0, len(Xp_test)): # If the patient has hepatitis C
            AST_upper = 31 if Xp_test[i,0] == 0 else 19 # Upper limit is 31 for men (0) and 19 for women
            AST = Xp_test[i,1]
            Plt = Xp_test[i,2]
            APRI = (100*AST/AST_upper)/(Plt)
            
            if (APRI >= 2):
                Yp_pred_new.append(4)
                Yp_test_new.append(Yp_test[i])
            elif (APRI <= 0.5):
                Yp_pred_new.append(0)
                Yp_test_new.append(Yp_test[i])

    cm = my_confusion_matrix(Yp_test_new,Yp_pred_new)
    print('Confusion matrix from inside APRI class!')
    print(cm)
    print('End of Confusion matrix from inside APRI class!')
    return Yp_pred_new, cm

def detapri_class (Xp_test, Yp_test):
    #Yp_pred = np.zeros([len(Yp_test),1]); # initialize to right size
    Yp_pred_new = [];
    Yp_test_new = [];
    
    for i in range(0, len(Xp_test)): # If the patient has hepatitis C
            AST_upper = 31 if Xp_test[i,0] == 0 else 19 # Upper limit is 31 for men (0) and 19 for women
            AST = Xp_test[i,1]
            Plt = Xp_test[i,2]
            APRI = (100*AST/AST_upper)/(Plt)
            
            if (APRI >= 2):
                Yp_pred_new.append(4)
                Yp_test_new.append(Yp_test[i])
            elif (APRI <= 0.5):
                Yp_pred_new.append(0)
                Yp_test_new.append(Yp_test[i])
            else:
                if (Yp_test[i] == 4):
                    Yp_pred_new.append(0)
                    Yp_test_new.append(Yp_test[i])
                else:
                    Yp_pred_new.append(4)
                    Yp_test_new.append(Yp_test[i])
                
        
    cm = my_confusion_matrix(Yp_test_new,Yp_pred_new)
    return Yp_pred_new, cm

def astalt_class (Xp_test, Yp_test):
    Yp_pred = [];
    
    for i in range(0, len(Xp_test)):
        Yp_pred.append(4 if (Xp_test[i,1]/Xp_test[i,0] >= 1) else 0)
    
    cm = my_confusion_matrix(Yp_test,Yp_pred)
    return Yp_pred, cm

def fib4_class (Xp_test, Yp_test):
    Yp_pred_new = [];
    Yp_test_new = [];

    for i in range(0, len(Xp_test)):
        age = Xp_test[i,0]
        ALT = Xp_test[i,1]
        AST = Xp_test[i,2]
        Plt = Xp_test[i,3]
        
        fib4 = age*AST/(Plt*(ALT)**0.5)
        
        if (fib4 <= 1.45):
            Yp_pred_new.append(0)
            Yp_test_new.append(Yp_test[i])
        elif (fib4 >= 3.25):
            Yp_pred_new.append(4)
            Yp_test_new.append(Yp_test[i])

        #print('age:       {} %'.format(age))
        #print('ALT:       {} %'.format(ALT))
        #print('AST:       {} %'.format(AST))
        #print('PLT:       {} %'.format(AST))
        #input('Continue?')
        
    
    cm = my_confusion_matrix(Yp_test_new,Yp_pred_new)
    #print(cm)
    #print(len(Yp_pred_new))
#    input('Press something')

    return Yp_pred_new, cm   

def detfib4_class (Xp_test, Yp_test):
    Yp_pred_new = [];
    Yp_test_new = [];

    for i in range(0, len(Xp_test)):
        age = Xp_test[i,0]
        ALT = Xp_test[i,1]
        AST = Xp_test[i,2]
        Plt = Xp_test[i,3]
        
        fib4 = age*AST/(Plt*(ALT)**0.5)
        
        if (fib4 <= 1.45):
            Yp_pred_new.append(0)
            Yp_test_new.append(Yp_test[i])
        elif (fib4 >= 3.25):
            Yp_pred_new.append(4)
            Yp_test_new.append(Yp_test[i])
        else:
            if (Yp_test[i] == 4):
                Yp_pred_new.append(0)
                Yp_test_new.append(Yp_test[i])
            else:
                Yp_pred_new.append(4)
                Yp_test_new.append(Yp_test[i])
                

        #print('age:       {} %'.format(age))
        #print('ALT:       {} %'.format(ALT))
        #print('AST:       {} %'.format(AST))
        #print('PLT:       {} %'.format(AST))
        #input('Continue?')
        
    
    cm = my_confusion_matrix(Yp_test_new,Yp_pred_new)
    #print(cm)
    #print(len(Yp_pred_new))
#    input('Press something')

    return Yp_pred_new, cm   

def calculate_metrics(cm, classifier, best_params_):
    tot = cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1] 
    tp = cm[1,1]
    fp = cm[0,1]
    tn = cm[0,0]
    fn = cm[1,0]
    accu = (tp + tn)/(tot)
    prec = tp/(tp + fp)
    reca = tp/(tp + fn)
    spec = tn/(tn + fp)
    falsePosRate = fp/(fp + tn)
    falseNegRate = fn/(fn + tp)
    f1 = 2*prec*reca/(prec+reca)
    print('')
    print('PLATELETS DATASET WITH {} CLASSIFIER:'.format(classifier))
    print(best_params_)
    print('F1:           {} %'.format(round(f1*100,2)))
    print('Sensitivity:  {} %'.format(round(reca*100,2)))
    print('Specificity:  {} %'.format(round(spec*100,2)))
    print('Precision     {} %'.format(round(prec*100,2)))
    print('Accuracy:     {} %'.format(round(accu*100,2)))
    print('FalsePosRate: {} %'.format(round(falsePosRate*100,2)))
    print('FalseNegRate: {} %'.format(round(falseNegRate*100,2)))
    print('')
    return f1,reca,spec,prec,accu,falsePosRate,falseNegRate

def create_correct_and_mis_excel_files(correct_mrn, mis_mrn_F4, mis_mrn_F01):
     data1 = pd.read_excel('C:/Users/Soren/Desktop/Thesis/Data Analysis/withPlatelets.xlsx')
     data1 = data1.loc[data1['MRNDate'].isin(correct_mrn)]
     data1 = (data1.style.applymap(lambda v: 'background-color: %s' % 'green' if 1==1 else 'green'))
     data1.to_excel('C:/Users/Soren/Desktop/Thesis/Data Analysis/withPlateletsCorrectlyClassified.xlsx', engine='openpyxl')
     
     data2 = pd.read_excel('C:/Users/Soren/Desktop/Thesis/Data Analysis/withPlatelets.xlsx')
     data2 = data2.loc[data2['MRNDate'].isin(mis_mrn_F4)]
     data2.to_excel('C:/Users/Soren/Desktop/Thesis/Data Analysis/withPlateletsMisclassifiedF4s.xlsx')
     
     data3 = pd.read_excel('C:/Users/Soren/Desktop/Thesis/Data Analysis/withPlatelets.xlsx')
     data3 = data3.loc[data3['MRNDate'].isin(mis_mrn_F01)]
     data3.to_excel('C:/Users/Soren/Desktop/Thesis/Data Analysis/withPlateletsMisclassifiedF01s.xlsx')     
     
def print_results(clf, args, feat, f1s, recs, specs, precs, accs, fns, fps, aucs):
    print(clf)
    print(args)
    print(feat, sep=',')
    print("F1:              %0.2f%%  (+/- %0.2f%%)" % (np.mean(f1s)*100, np.std(f1s)*100)) 
    print("Sensitivity:     %0.2f%%  (+/- %0.2f%%)" % (np.mean(recs)*100, np.std(recs)*100)) 
    print("Specificity:     %0.2f%%  (+/- %0.2f%%)" % (np.mean(specs)*100, np.std(specs)*100))  
    print("Precision:       %0.2f%%  (+/- %0.2f%%)" % (np.mean(precs)*100, np.std(precs)*100))   
    print("Accuracy:        %0.2f%%  (+/- %0.2f%%)" % (np.mean(accs)*100, np.std(accs)*100)) 
    print("False Neg Rate:  %0.2f%%  (+/- %0.2f%%)" % (np.mean(fns)*100, np.std(fns)*100))
    print("False Pos Rate:  %0.2f%%  (+/- %0.2f%%)" % (np.mean(fps)*100, np.std(fps)*100))
    print("AUROC:           %0.2f%%  (+/- %0.2f%%)" % (np.mean(aucs)*100, np.std(aucs)*100)) 
    print()     
    print("Cost function score: %0.2f%%" % ((np.mean(recs)*100 + np.mean(accs)*100 + np.mean(aucs)*100)/3))
    print()     

def find_misclassified_patients(m1, m2, m3, m4, m5, m6, m7, ex, MRNDates):
    correct_mrns = list()
    misclassified_F01s = list()
    misclassified_F4s = list()
    j = 0;
    print('High Risk Misclassifications: Identified as F0 or F1 but actually F4')
    for i in range(0,np.size(m1)):
        if ((m1[i] != ex[i]) and (m2[i] != ex[i]) and (m3[i] != ex[i]) and (m4[i] != ex[i]) and (m5[i] != ex[i]) and (m6[i] != ex[i]) and (m7[i] != ex[i]) and ex[i] == 4):
            j = j + 1;
            print(MRNDates[i]) 
            misclassified_F4s.append(MRNDates[i])
    print('')
    print('Low Risk Misclassifications: Identified as F4 but actually F0 or F1')
    for i in range(0,np.size(m1)):
        if ((m1[i] != ex[i]) and (m2[i] != ex[i]) and (m3[i] != ex[i]) and (m4[i] != ex[i]) and (m5[i] != ex[i]) and (m6[i] != ex[i]) and (m7[i] != ex[i]) and ex[i] == 0):
            j = j + 1;
            print(MRNDates[i]) 
            misclassified_F01s.append(MRNDates[i])
    print('')
    for i in range(0,np.size(m1)):
        if ((m1[i] == ex[i]) or (m2[i] == ex[i]) or (m3[i] == ex[i]) or (m4[i] == ex[i]) or (m5[i] == ex[i]) or (m6[i] == ex[i]) or (m7[i] == ex[i])):
            j = j + 1;    
            #print(MRNDates(i)) 
            correct_mrns.append(MRNDates[i])
    print('Number of misclassified F4s: {}'.format(np.size(misclassified_F4s)))
    print('Number of misclassified F01s: {}'.format(np.size(misclassified_F01s)))
    print('Number of correct classifications: {}'.format(np.size(correct_mrns)))

    return correct_mrns, misclassified_F4s, misclassified_F01s

def plot_heat_map(method1,label1, method2, label2, method3, label3,method4,label4,method5,label5,method6,label6,method7,label7, exact, directory, name):
    a = np.random.randn(8,np.size(method1))
    for i in range (0, np.size(method1,0) -1):
            a[0,i] = method1[i]
            a[1,i] = method2[i]
            a[2,i] = method3[i]
            a[3,i] = method4[i]
            a[4,i] = method5[i]
            a[5,i] = method6[i]
            a[6,i] = method7[i]
            a[7,i] = exact[i]
            
            if (a[0,i] != a[7,i] and a[1,i] != a[7,i] and a[2,i] != a[7,i] and a[3,i] != a[7,i] and a[4,i] != a[7,i] and a[5,i] != a[7,i] and a[6,i] != a[7,i]):
                print(i)
                a[7,i] = 8;
    
    df = pd.DataFrame(a)
    # Values from 0-2 will be green and 2-5 will be red
    import matplotlib.colors as mcolors
    cmap, norm = mcolors.from_levels_and_colors([0, 2, 4.1, 9], ['lightblue', 'yellow', 'red'])
    plt.pcolor(df, cmap=cmap, norm=norm)
    plt.yticks(np.arange(0.5, len(df.index), 1), [label1,label2,label3,label4,label5,label6,label7,'Actual'])
    plt.ylabel('Algorithm')
    plt.xlabel('Record Number')
    plt.savefig(directory + name + '_HeatMap.png')
    plt.show()
    
def rank_feature_importance(name, fi, feats):
    imp = np.zeros((np.size(fi),2))
    imp[:,0] = fi;
    
    for i in range(0,np.size(imp,0)):
        imp[i,1] = i
    
    imp = (sorted(imp, key=lambda x: x[0]))
    
    features = get_features(feats)

    for i in range (np.size(imp,0)-1,-1,-1):
        print('%s  %0.3f' % (features[imp[i][1].astype(int)].ljust(25),  (imp[i][0]))) 
        #print('{}|   {}'.format(features[imp[i][1].astype(int)].ljust(16), (imp[i][0])) 
    print(feats)
    return 

def get_transformations(Xp):
    Xp_t = np.zeros((np.size(Xp,0), np.size(Xp,1)))
    
    Sex_p = Xp[:,0] # No Transformations
    Xp_t[:,0] = Xp[:,0]
    
    Age_p = Xp[:,1] # No Transformations
    Xp_t[:,1] = Age_p

    Albumin_p = Xp[:,2] # Use Cube Transformation
    Xp_t[:,2] =Albumin_p**3.025
    
    ALP_p = Xp[:,3] # Use Inverse and Power Transform
    Xp_t[:,3] = 1/(ALP_p**0.8)
    
    ALT_p = Xp[:,4] # Use Inverse and Power Transform
    Xp_t[:,4] = 1/(ALT_p**.71125)
    
    AST_p = Xp[:,5] # Inverse Log Power Transform
    Xp_t[:,5] = 1/(np.log(AST_p))**3.5
    
    Bilirubin_p = Xp[:,6] # Inverse Log Power Transform
    Xp_t[:,6] = 1/np.log(Bilirubin_p)**0.001
    
    Creatinine_p = Xp[:,7] # Inverse Power Transform
    Xp_t[:,7] = 1/Creatinine_p**.25
    
    INR_p = Xp[:,8] # Inverse Power Transform
    Xp_t[:,8] = 1/(INR_p**5.1)
    
    BMI_p = Xp[:,9] # No Transformation 
    Xp_t[:,9] = BMI_p    
    
    Platelets_p = Xp[:,10] # No Transfoormation
    Xp_t[:,10] = Platelets_p
    
    Xp_t[:,11:] = Xp[:,11:] # All other variables are binary and do not need transforming
    return Xp_t

def print_guesses(name, cm, pred, truth):
    print()
    print('From print guesses: ')
    print()
    
    TN_count = 0
    TP_count = 0
    FP_count = 0
    FN_count = 0
    
    for i in range(0, len(pred)):
        if (pred[i] == 0 and truth[i] == 0):
            print(str(i) + ". " + name + " guessed: " + str(pred[i]) + " but it was " + str(truth[i]) + "   TN")
            TN_count += 1
        elif(pred[i] == 4 and truth[i] == 4):
            print(str(i) + ". " + name + " guessed: " + str(pred[i]) + " but it was " + str(truth[i]) + "   TP")
            TP_count += 1
        elif(pred[i] == 4 and truth[i] == 0):
            print(str(i) + ". " + name + " guessed: " + str(pred[i]) + " but it was " + str(truth[i]) + "   FP")
            FP_count += 1
        elif(pred[i] == 0 and truth[i] == 4):
            print(str(i) + ". " + name + " guessed: " + str(pred[i]) + " but it was " + str(truth[i]) + "   FN")
            FN_count += 1
            
#        print('Tally from print_guesses at: ')
#        print('TP Count: ' + str(TP_count))
#        print('TN Count: ' + str(TN_count))       
#        print('FP Count: ' + str(FP_count))        
#        print('FN Count: ' + str(FN_count))      
#        print(cm)
#        input('Superman')
    
#svm_pred, rfc_pred, gbc_pred, log_pred, knn_pred, mlp_pred, ens_pred, astalt_pred, apri_pred, fib4_pred = validate_performance(Xp[:,sv], Yp[:,0].astype(int), Xp_val, Yp_val.values.astype(int))

def my_confusion_matrix(truth, pred):
    TN_count = 0
    TP_count = 0
    FP_count = 0
    FN_count = 0
    
    for i in range(0, len(pred)):
        if (pred[i] == 0 and truth[i] == 0):
            print(str(i) + ". " + " guessed: " + str(pred[i]) + " but it was " + str(truth[i]) + "   TN")
            TN_count += 1
        elif(pred[i] == 4 and truth[i] == 4):
            print(str(i) + ". "  + " guessed: " + str(pred[i]) + " but it was " + str(truth[i]) + "   TP")
            TP_count += 1
        elif(pred[i] == 4 and truth[i] == 0):
            print(str(i) + ". "+ " guessed: " + str(pred[i]) + " but it was " + str(truth[i]) + "   FP")
            FP_count += 1
        elif(pred[i] == 0 and truth[i] == 4):
            print(str(i) + ". " + " guessed: " + str(pred[i]) + " but it was " + str(truth[i]) + "   FN")
            FN_count += 1
        
        cm = np.ndarray(shape=(2,2))
        cm[1,1] = TP_count 
        cm[0,1] = FP_count
        cm[0,0] = TN_count
        cm[1,0] = FN_count
#        print('Tally from inside my_confusion_Matrix: ')
#        print('TP Count: ' + str(TP_count))
#        print('TN Count: ' + str(TN_count))       
#        print('FP Count: ' + str(FP_count))        
#        print('FN Count: ' + str(FN_count))      
        print(cm)
#        input('Batman!')
    
            
#    print('True positive count from inside my_cm: ' + str(TP_count))
#    print('True negative count from inside my_cm: ' + str(TN_count))
#    print('False positive count from inside my_cm: ' + str(FP_count))
#    print('False negative count from inside my_cm: ' + str(FN_count))
             
    cm = np.ndarray(shape=(2,2))
    cm[1,1] = TP_count 
    cm[0,1] = FP_count
    cm[0,0] = TN_count
    cm[1,0] = FN_count
#    print(cm)
#    print('True positive count from inside my_cm: ' + str(TP_count))
#    print('True negative count from inside my_cm: ' + str(TN_count))
#    print('False positive count from inside my_cm: ' + str(FP_count))
#    print('False negative count from inside my_cm: ' + str(FN_count))
    return cm

df = pd.read_excel('C:/Users/Soren/Desktop/Thesis/Data Analysis/dataset.xlsx', parse_cols="A:AT")
#df = df.loc[(df['Platelets'] >= 0) & (df['TotalMissingnessWithPlatelets'] < 0.4)]
#df = df.loc[(df['Platelets'] >= 0) & (df['TotalMissingnessWithPlatelets'] < 0.4) & (df['DecompensatedCirrhosis'] == 0)]
df = df.loc[(df['TotalMissingnessWithPlatelets'] < 0.4) & (df['DecompensatedCirrhosis'] == 0) & (df['NAFL'] == 1)]
df = df.sample(frac=1).reset_index(drop=True)
Xp = df.iloc[:,0:39].values
Yp = df.iloc[:,[39,41]].values 
Yp[:,0] = nd.replace(nd.replace(nd.replace(Yp[:,0].astype(str), 'F 4', '4'), 'F 1', '0'), 'F 0', '0').astype(int)
groups = df.iloc[:,41].astype(int)

from sklearn.model_selection import GroupKFold
kf = GroupKFold(n_splits=10)
kf.get_n_splits(Xp, Yp, groups)

# List of features
# 0: Sex
# 1: Age
# 2: Albumin
# 3: ALP
# 4: ALT
# 5: AST
# 6: Bilirubin
# 7: Creatinine
# 8: INR
# 9: BMI
# 10: Platelets
# 20: Diabetes

sv = [0,1,2,4,5,6,8,9,10,20]; #21,22,23,24,25,26
rf = [0,1,2,4,5,6,8,9,10,20]; #21,22,23,24,25,26
gb = [0,1,2,4,5,6,8,9,10,20]; #11
lg = [0,1,2,4,5,6,8,9,10,20]; #11,13
kn = [0,1,2,4,5,6,8,9,10,20]; #11
ml = [0,1,2,4,5,6,8,9,10,20]; 
apr = [0,5,10]; # Sex, AST, and Platelets, and Hepatitis C 
asal = [4,5]; # ALT, AST
fb4 = [1,4,5,10] # Age, ALT, AST, Platelets

#svm_preds, rfc_preds, gbc_preds, log_preds, knn_preds, mlp_preds, ens_preds,astalt_preds, fib4_preds, apri_preds, Yp_tests, Yp_tests_MRNDates, metrics, errors = run_everything('old', Xp, Yp, groups)

# Now, import the validation set and compare numbers. 
val_df = pd.read_excel('C:/Users/Soren/Desktop/Thesis/Data Analysis/mcgill_dataset.xlsx', parse_cols="A:R")
val_df = val_df.loc[((val_df['updatedFib'] == 0) | (val_df['updatedFib'] == 4)) & (val_df['Missingness'] <= 3) & (val_df['Etiology']==1)]
values = {'DM': 0}
val_df = val_df.fillna(value=values) 
Xp_val = val_df.iloc[:,0:10]
Yp_val = val_df.iloc[:,10]

#, rfc_pred, gbc_pred, log_pred, knn_pred, mlp_pred
def validate_performance(Xp_toronto, Yp_toronto, Xp_mcgill, Yp_mcgill):
    #Xp_toronto = MICE().complete(Xp_toronto)
    #Xp_mcgill = MICE().complete(Xp_mcgill)
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import Imputer
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    Xp_toronto = imp.fit_transform(Xp_toronto)
    Xp_mcgill = imp.fit_transform(Xp_mcgill)
    
    Xp_mcgill_unscaled = Xp_mcgill
    
    sc_p = StandardScaler()
    Xp_toronto = sc_p.fit_transform(Xp_toronto)    
    Xp_mcgill = sc_p.transform(Xp_mcgill)
    
    svm_pred=[]; rfc_pred=[]; gbc_pred=[]; log_pred=[]; knn_pred=[]; mlp_pred=[]; ens_pred=[]; astalt_pred=[]; apri_pred=[]; fib4_pred=[];
    
#    svm_pred, svm_cm, svm_proba, svm_args = svm_class(Xp_toronto, Xp_mcgill, Yp_toronto, Yp_mcgill)
#    svm_auc = (roc_auc_score(Yp_mcgill.astype(int)/4, svm_proba))
#    svm_f1,svm_reca,svm_spec, svm_prec,svm_accu,svm_fp,svm_fn = calculate_metrics(svm_cm, 'svm', None)  
#    print("Auroc: %0.2f%%" % (svm_auc*100).astype(float))
#    
#    generate_graph=False
#    rfc_pred, rfc_cm, rfc_importances, rfc_proba, rfc_args = rfc_class(Xp_toronto, Xp_mcgill, Yp_toronto, Yp_mcgill, generate_graph)
#    rfc_auc =(roc_auc_score(Yp_mcgill.astype(int)/4, rfc_proba))
#    rfc_f1,rfc_reca,rfc_spec,rfc_prec,rfc_accu,rfc_fp,rfc_fn = calculate_metrics(rfc_cm, 'rfc', None)
#    #best_rfc_feature = rank_feature_importance("RFC", rfc_importances, rf)
#    print("Auroc: %0.2f%%" % (rfc_auc*100))
#    
#    gbc_pred, gbc_cm, gbc_proba, gbc_args = gbc_class(Xp_toronto, Xp_mcgill, Yp_toronto, Yp_mcgill)
#    gbc_f1,gbc_reca,gbc_spec,gbc_prec,gbc_accu,gbc_fp,gbc_fn = calculate_metrics(gbc_cm, 'gbc', None)
#    gbc_auc = (roc_auc_score(Yp_mcgill.astype(int)/4, gbc_proba))
#    print("Auroc: %0.2f%%" % (gbc_auc*100))
#    
#    log_pred, log_cm, log_proba, log_args = log_class(Xp_toronto, Xp_mcgill, Yp_toronto, Yp_mcgill)
#    log_f1,log_reca,log_spec,log_prec,log_accu,log_fp,log_fn = calculate_metrics(log_cm, 'log', None)
#    log_auc = (roc_auc_score(Yp_mcgill.astype(int)/4, log_proba))
#    print("Auroc: %0.2f%%" % (log_auc*100))
#    
#    knn_pred, knn_cm, knn_proba, knn_args = knn_class(Xp_toronto, Xp_mcgill, Yp_toronto, Yp_mcgill)
#    knn_f1,knn_reca,knn_spec,knn_prec,knn_accu,knn_fp,knn_fn = calculate_metrics(knn_cm, 'knn', None)
#    knn_auc = (roc_auc_score(Yp_mcgill.astype(int)/4, knn_proba))
#    print("Auroc: %0.2f%%" % (knn_auc*100))
#    
#    mlp_pred, mlp_cm, mlp_proba, mlp_args = mlp_class(Xp_toronto, Xp_mcgill, Yp_toronto, Yp_mcgill)
#    mlp_f1,mlp_reca,mlp_spec,mlp_prec,mlp_accu,mlp_fp,mlp_fn = calculate_metrics(mlp_cm, 'mlp', None)
#    mlp_auc = (roc_auc_score(Yp_mcgill.astype(int)/4, mlp_proba))   
#    print("Auroc: %0.2f%%" % (mlp_auc*100))
#    
#    ens_pred, ens_cm, ens_proba, ens_args = ens_class(svm_proba,rfc_proba,gbc_proba,log_proba,knn_proba, mlp_proba, Yp_mcgill, 0.5)
#    ens_f1,ens_reca,ens_spec,ens_prec,ens_accu,ens_fp,ens_fn = calculate_metrics(ens_cm, 'ens', None)
#    ens_auc = (roc_auc_score(Yp_mcgill.astype(int)/4, ens_proba))
#    print("Auroc: %0.2f%%" % (ens_auc*100))

#    astalt_pred, astalt_cm = astalt_class(Xp_mcgill_unscaled[:,[3,4]], Yp_mcgill)
#    astalt_f1,astalt_reca,astalt_spec,astalt_prec,astalt_accu,astalt_fp,astalt_fn = calculate_metrics(astalt_cm, 'astalt', None)
#    
    apri_pred, apri_cm = apri_class(Xp_mcgill_unscaled[:,[0,4,8]], Yp_mcgill);
    apri_f1,apri_reca,apri_spec,apri_prec,apri_accu,apri_fp,apri_fn = calculate_metrics(apri_cm, 'apri', None)
    print(apri_cm)
#    fib4_pred, fib4_cm = fib4_class(Xp_mcgill_unscaled[:,[1,3,4,8]], Yp_mcgill)
#    fib4_f1,fib4_reca,fib4_spec,fib4_prec,fib4_accu,fib4_fp,fib4_fn = calculate_metrics(fib4_cm, 'fib4', None)

    print(str(len(apri_pred)))
    print(str(len(Yp_mcgill)))

    print_guesses('apri', apri_cm, apri_pred, Yp_mcgill)
#    print_guesses('fib4', fib4_cm, fib4_pred, Yp_mcgill)

#
#    from beautifultable import BeautifulTable
#    table = BeautifulTable(max_width=300)
#    table.column_headers = [" ", "SVM", "RFC", "GBC","LOG","KNN","NN","ENS", "AST/ALT", "FIB4", "APRI"]
#    table.append_row(['F1' ,('%.2f' % (np.mean(svm_f1)*100)),('%.2f' % (np.mean(rfc_f1)*100)) ,('%.2f' % (np.mean(gbc_f1)*100)),('%.2f' % (np.mean(log_f1)*100)) ,('%.2f' % (np.mean(knn_f1)*100)),('%.2f' % (np.mean(mlp_f1)*100)),('%.2f' % (np.mean(ens_f1)*100)), ('%.2f' % (np.mean(astalt_f1)*100)),('%.2f' % (np.mean(fib4_f1)*100)),('%.2f' % (np.mean(apri_f1)*100))])
#    table.append_row(['Sensitivity',('%.2f' % (np.mean(svm_reca)*100)),('%.2f' % (np.mean(rfc_reca)*100)),('%.2f' % (np.mean(gbc_reca)*100)),('%.2f' % (np.mean(log_reca)*100)) ,('%.2f' % (np.mean(knn_reca)*100)) ,('%.2f' % (np.mean(mlp_reca)*100)) ,('%.2f' % (np.mean(ens_reca)*100)) ,('%.2f' % (np.mean(astalt_reca)*100)) ,('%.2f' % (np.mean(fib4_reca)*100)), ('%.2f' % (np.mean(apri_reca)*100))])
#    table.append_row(['Specificity',('%.2f' % (np.mean(svm_spec)*100)),('%.2f' % (np.mean(rfc_spec)*100)),('%.2f' % (np.mean(gbc_spec)*100)),('%.2f' % (np.mean(log_spec)*100)) ,('%.2f' % (np.mean(knn_spec)*100)) ,('%.2f' % (np.mean(mlp_spec)*100)) ,('%.2f' % (np.mean(ens_spec)*100)) ,('%.2f' % (np.mean(astalt_spec)*100)) ,('%.2f' % (np.mean(fib4_spec)*100)), ('%.2f' % (np.mean(apri_spec)*100))])    
#    table.append_row(['Precision',('%.2f' % (np.mean(svm_prec)*100)),('%.2f' % (np.mean(rfc_prec)*100)),('%.2f' % (np.mean(gbc_prec)*100)),('%.2f' % (np.mean(log_prec)*100)),('%.2f' % (np.mean(knn_prec)*100)),('%.2f' % (np.mean(mlp_prec)*100)),('%.2f' % (np.mean(ens_prec)*100)) ,('%.2f' % (np.mean(astalt_prec)*100)), ('%.2f' % (np.mean(fib4_prec)*100)), ('%.2f' % (np.mean(apri_prec)*100))])
#    table.append_row(['Accuracy',('%.2f' % (np.mean(svm_accu)*100)),('%.2f' % (np.mean(rfc_accu)*100)),('%.2f' % (np.mean(gbc_accu)*100)),('%.2f' % (np.mean(log_accu)*100)),('%.2f' % (np.mean(knn_accu)*100)),('%.2f' % (np.mean(mlp_accu)*100)),('%.2f' % (np.mean(ens_accu)*100)) ,('%.2f' % (np.mean(astalt_accu)*100)), ('%.2f' % (np.mean(fib4_accu)*100)), ('%.2f' % (np.mean(apri_accu)*100))])
#    table.append_row(['False Neg Rate',('%.2f' % (np.mean(svm_fn)*100)),('%.2f' % (np.mean(rfc_fn)*100)),('%.2f' % (np.mean(gbc_fn)*100)),('%.2f' % (np.mean(log_fn)*100)),('%.2f' % (np.mean(knn_fn)*100)),('%.2f' % (np.mean(mlp_fn)*100)),('%.2f' % (np.mean(ens_fn)*100)) ,('%.2f' % (np.mean(astalt_fn)*100)), ('%.2f' % (np.mean(fib4_fn)*100)), ('%.2f' % (np.mean(apri_fn)*100))])
#    table.append_row(['False Pos Rate',('%.2f' % (np.mean(svm_fp)*100)),('%.2f' % (np.mean(rfc_fp)*100)),('%.2f' % (np.mean(gbc_fp)*100)),('%.2f' % (np.mean(log_fp)*100)),('%.2f' % (np.mean(knn_fp)*100)),('%.2f' % (np.mean(mlp_fp)*100)),('%.2f' % (np.mean(ens_fp)*100)) ,('%.2f' % (np.mean(astalt_fp)*100)), ('%.2f' % (np.mean(fib4_fp)*100)), ('%.2f' % (np.mean(apri_fp)*100))])
#    table.append_row(['Auroc',('%.2f' % (np.mean(svm_auc)*100)),('%.2f' % (np.mean(rfc_auc)*100)),('%.2f' % (np.mean(gbc_auc)*100)),('%.2f' % (np.mean(log_auc)*100)),('%.2f' % (np.mean(knn_auc)*100)),('%.2f' % (np.mean(mlp_auc)*100)),('%.2f' % (np.mean(ens_auc)*100)) ,('NaN'), ('NaN'), ('NaN')])
#    print(table)

    return svm_pred, rfc_pred, gbc_pred, log_pred, knn_pred, mlp_pred, ens_pred, astalt_pred, apri_pred, fib4_pred#, rfc_pred, gbc_pred, log_pred, knn_pred, mlp_pred
  
svm_pred, rfc_pred, gbc_pred, log_pred, knn_pred, mlp_pred, ens_pred, astalt_pred, apri_pred, fib4_pred = validate_performance(Xp[:,sv], Yp[:,0].astype(int), Xp_val, Yp_val.values.astype(int))
