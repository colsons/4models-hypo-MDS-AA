# 4models-hypo-MDS-AA
hypo-MDS+AA four models of disease diagnosis（model：svm，bayes，cnn，Decision Tree）

此诊断模型部署在阿里云服务器，
可以通过[colsons.site/disease/index/](colsons.site/disease/index/)访问
（结果作为智能辅助诊断结果，并非绝对严谨）

data中存放数据集
log为cnn模型的日志结果
res为cnn模型loaddown

evel.py为评估系数计算文件
*—model.py/ipynb 为模型运行文件
SVM.py与util为svm_model.py前置函数类
