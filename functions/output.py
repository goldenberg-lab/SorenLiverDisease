import numpy as np
from beautifultable import BeautifulTable

def print_table(objArray, uncertainty):
    headers = [" "]
    f1Row = ['F1']
    sensRow = ['Sensitivity']
    specsRow = ['Specificity']
    precsRow = ['Precision']
    accsRow = ['Accuracy']
    fnsRow = ['False Neg Rate']
    fpsRow = ['False Pos Rate']
    aucRow = ['AUROC']
    table_width = 0
    oc = 1
    
    for obj in objArray:
        headers.append(obj.name)
        if (uncertainty == True):
            f1Row.append(((np.mean(obj.f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(obj.f1s)*100)))
            sensRow.append(((np.mean(obj.sens)*100)) + ' +/- ' + ('%.2f' % (np.std(obj.sens)*100)))
            specsRow.append(((np.mean(obj.specs)*100)) + ' +/- ' + ('%.2f' % (np.std(obj.specs)*100)))
            precsRow.append(((np.mean(obj.precs)*100)) + ' +/- ' + ('%.2f' % (np.std(obj.precs)*100)))
            accsRow.append(((np.mean(obj.accs)*100)) + ' +/- ' + ('%.2f' % (np.std(obj.accs)*100)))
            fnsRow.append(((np.mean(obj.fns)*100)) + ' +/- ' + ('%.2f' % (np.std(obj.fns)*100)))
            fpsRow.append(((np.mean(obj.fps)*100)) + ' +/- ' + ('%.2f' % (np.std(obj.fps)*100)))
            aucRow.append(((np.mean(obj.aucs)*100)) + ' +/- ' + ('%.2f' % (np.std(obj.aucs)*100)))
        else:
            f1Row.append((np.mean(obj.f1s)*100))
            sensRow.append((np.mean(obj.sens)*100))
            specsRow.append((np.mean(obj.specs)*100))
            precsRow.append((np.mean(obj.precs)*100))
            accsRow.append((np.mean(obj.accs)*100))
            fnsRow.append((np.mean(obj.fns)*100))
            fpsRow.append((np.mean(obj.fps)*100))
            aucRow.append((np.mean(obj.aucs)*100))
        table_width += max(len(f1Row[oc]),len(sensRow[oc]), len(specsRow[oc]), len(precsRow[oc]), len(accsRow[oc]), len(fnsRow[oc]), len(fpsRow[oc]), len(aucRow[oc])) + 2
       
    
    table = BeautifulTable(max_width=table_width)
    table.column_headers = headers
    table.append_row(f1Row)
    table.append_row(sensRow)
    table.append_row(specsRow)
    table.append_row(precsRow)
    table.append_row(accsRow)
    table.append_row(fnsRow)
    table.append_row(fpsRow)
    table.append_row(aucRow)
    print(table)


#    table.column_headers = [" ", "SVM", "RFC", "GBC","LOG","KNN","MLP","ENS", "AST/ALT", "FIB4", "APRI"]
#    table.append_row(['F1' ,('%.2f' % (np.mean(svm_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(svm_f1s)*100)),('%.2f' % (np.mean(rfc_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(rfc_f1s)*100)),('%.2f' % (np.mean(gbc_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(gbc_f1s)*100)),('%.2f' % (np.mean(log_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(log_f1s)*100)),('%.2f' % (np.mean(knn_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(knn_f1s)*100)),('%.2f' % (np.mean(mlp_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(mlp_f1s)*100)),('%.2f' % (np.mean(ens_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_f1s)*100)), ('%.2f' % (np.mean(astalt_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_f1s)*100)),('%.2f' % (np.mean(fib4_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_f1s)*100)),('%.2f' % (np.mean(apri_f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_f1s)*100))])
#    table.append_row(['Sensitivity',('%.2f' % (np.mean(svm_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(svm_recs)*100)),('%.2f' % (np.mean(rfc_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(rfc_recs)*100)),('%.2f' % (np.mean(gbc_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(gbc_recs)*100)),('%.2f' % (np.mean(log_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(log_recs)*100)),('%.2f' % (np.mean(knn_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(knn_recs)*100)),('%.2f' % (np.mean(mlp_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(mlp_recs)*100)),('%.2f' % (np.mean(ens_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_recs)*100)),('%.2f' % (np.mean(astalt_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_recs)*100)),('%.2f' % (np.mean(fib4_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_recs)*100)), ('%.2f' % (np.mean(apri_recs)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_recs)*100))])
#    table.append_row(['Specificty',('%.2f' % (np.mean(svm_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(svm_specs)*100)),('%.2f' % (np.mean(rfc_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(rfc_specs)*100)),('%.2f' % (np.mean(gbc_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(gbc_specs)*100)),('%.2f' % (np.mean(log_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(log_specs)*100)),('%.2f' % (np.mean(knn_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(knn_specs)*100)),('%.2f' % (np.mean(mlp_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(mlp_specs)*100)),('%.2f' % (np.mean(ens_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_specs)*100)),('%.2f' % (np.mean(astalt_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_specs)*100)),('%.2f' % (np.mean(fib4_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_specs)*100)), ('%.2f' % (np.mean(apri_specs)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_specs)*100))])    
#    table.append_row(['Precision',('%.2f' % (np.mean(svm_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(svm_precs)*100)),('%.2f' % (np.mean(rfc_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(rfc_precs)*100)),('%.2f' % (np.mean(gbc_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(gbc_precs)*100)),('%.2f' % (np.mean(log_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(log_precs)*100)),('%.2f' % (np.mean(knn_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(knn_precs)*100)),('%.2f' % (np.mean(mlp_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(mlp_precs)*100)),('%.2f' % (np.mean(ens_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_precs)*100)),('%.2f' % (np.mean(astalt_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_precs)*100)),('%.2f' % (np.mean(fib4_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_precs)*100)),('%.2f' % (np.mean(apri_precs)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_precs)*100))])
#    table.append_row(['Accuracy',('%.2f' % (np.mean(svm_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(svm_accs)*100)),('%.2f' % (np.mean(rfc_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(rfc_accs)*100)),('%.2f' % (np.mean(gbc_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(gbc_accs)*100)),('%.2f' % (np.mean(log_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(log_accs)*100)),('%.2f' % (np.mean(knn_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(knn_accs)*100)),('%.2f' % (np.mean(mlp_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(mlp_accs)*100)),('%.2f' % (np.mean(ens_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_accs)*100)),('%.2f' % (np.mean(astalt_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_accs)*100)),('%.2f' % (np.mean(fib4_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_accs)*100)), ('%.2f' % (np.mean(apri_accs)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_accs)*100))])
#    table.append_row(['False Neg Rate',('%.2f' % (np.mean(svm_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(svm_fns)*100)),('%.2f' % (np.mean(rfc_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(rfc_fns)*100)),('%.2f' % (np.mean(gbc_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(gbc_fns)*100)),('%.2f' % (np.mean(log_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(log_fns)*100)),('%.2f' % (np.mean(knn_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(knn_fns)*100)), ('%.2f' % (np.mean(mlp_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(mlp_fns)*100)),('%.2f' % (np.mean(ens_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_fns)*100)),('%.2f' % (np.mean(astalt_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_fns)*100)),('%.2f' % (np.mean(fib4_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_fns)*100)), ('%.2f' % (np.mean(apri_fns)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_fns)*100))])
#    table.append_row(['False Pos Rate',('%.2f' % (np.mean(svm_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(svm_fps)*100)),('%.2f' % (np.mean(rfc_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(rfc_fps)*100)),('%.2f' % (np.mean(gbc_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(gbc_fps)*100)),('%.2f' % (np.mean(log_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(log_fps)*100)),('%.2f' % (np.mean(knn_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(knn_fps)*100)), ('%.2f' % (np.mean(mlp_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(mlp_fps)*100)) ,('%.2f' % (np.mean(ens_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(ens_fps)*100)),('%.2f' % (np.mean(astalt_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(astalt_fps)*100)),('%.2f' % (np.mean(fib4_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(fib4_fps)*100)), ('%.2f' % (np.mean(apri_fps)*100)) + ' +/- ' + ('%.2f' % (np.std(apri_fps)*100))])
#    table.append_row(['AUROC',('%.2f' % (np.mean(svm_aucs))) + ' +/- ' + ('%.2f' % (np.std(svm_aucs))),('%.2f' % (np.mean(rfc_aucs))) + ' +/- ' + ('%.2f' % (np.std(rfc_aucs))),('%.2f' % (np.mean(gbc_aucs))) + ' +/- ' + ('%.2f' % (np.std(gbc_aucs))),('%.2f' % (np.mean(log_aucs))) + ' +/- ' + ('%.2f' % (np.std(log_aucs))),('%.2f' % (np.mean(knn_aucs))) + ' +/- ' + ('%.2f' % (np.std(knn_aucs))),('%.2f' % (np.mean(mlp_aucs))) + ' +/- ' + ('%.2f' % (np.std(mlp_aucs))),('%.2f' % (np.mean(ens_aucs))) + ' +/- ' + ('%.2f' % (np.std(ens_aucs))),("NaN +/- NaN"),("NaN +/- NaN"), ("NaN +/- NaN")])
#    print(table)