# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:22:41 2022

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 15:29:11 2021

@author: user
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from typing import List, Dict
from sklearn import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
class feature_extractor(nn.Module):
    def __init__(self,hidden_1,hidden_2):
         super(feature_extractor,self).__init__()
         self.fc1=nn.Linear(310,hidden_1)
         self.fc2=nn.Linear(hidden_1,hidden_2)
         self.dropout1 = nn.Dropout(p=0.25)
         self.dropout2 = nn.Dropout(p=0.25)
    def forward(self,x):
         x=self.fc1(x)
         x=F.relu(x)
#         x=F.leaky_relu(x)
         x=self.fc2(x)
         x=F.relu(x)
#         x=F.leaky_relu(x)
         return x
    def get_parameters(self) -> List[Dict]:
         params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
                  ]
         return params  
class discriminator(nn.Module):
    def __init__(self,hidden_1):
         super(discriminator,self).__init__()
         self.fc1=nn.Linear(hidden_1,hidden_1)
         self.fc2=nn.Linear(hidden_1,1)
         self.dropout1 = nn.Dropout(p=0.25)
         self.sigmoid = nn.Sigmoid()
    def forward(self,x):
         x=self.fc1(x)
         x=F.relu(x)
#         x=F.leaky_relu(x)
         x=self.dropout1(x)
         x=self.fc2(x)
         x=self.sigmoid(x)
         return x
    def get_parameters(self) -> List[Dict]:
         params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
                  ]
         return params 
class Domain_adaption_model(nn.Module):
   def __init__(self,hidden_1,hidden_2,hidden_3,hidden_4,num_of_class,low_rank,max_iter,upper_threshold,lower_threshold):
       super(Domain_adaption_model,self).__init__()
       self.fea_extrator_f= feature_extractor(hidden_1,hidden_2)
       self.fea_extrator_g= feature_extractor(hidden_3,hidden_4)
       self.U=nn.Parameter(torch.randn(low_rank,hidden_2),requires_grad=True)
       self.V=nn.Parameter(torch.randn(low_rank,hidden_4),requires_grad=True)
       self.P=torch.randn(num_of_class,hidden_4)
       self.stored_mat=torch.matmul(self.V,self.P.T)
       self.max_iter=max_iter
       self.upper_threshold=upper_threshold
       self.lower_threshold=lower_threshold
#       self.diff=(upper_threshold-lower_threshold)
       self.threshold=upper_threshold
       self.cluster_label=np.zeros(num_of_class)#这个变量装的是映射关系，就是聚类0,1,2对应的标签
       self.num_of_class=num_of_class
   def forward(self,source,target,source_label):
       feature_source_f=self.fea_extrator_f(source)
       feature_target_f=self.fea_extrator_f(target)
#       feature_source_g=feature_source_f
       feature_source_g=self.fea_extrator_f(source)
       ##torch.matmul(source_label.T,torch.ones(batch_num,num_of_class))
       ## Update P through some algebra computations for the convenice of broadcast

# 求原型特征，原型特征是相同情绪类别的累加后求平均，下面的P变量就是原型特征
# 解释：对one-hot编码的source_label样本维度求和后，再对角是统计各个情绪类别的出现次数
# 解释：加1求逆是直接求倒数，后面矩阵乘法source_label @ feature就是对不同情绪类别的标签求和
# 最终原型特征就是每种情绪下的特征求平均，最后P变成了一个3（情绪类别）*64（特征维度）的矩阵

       self.P=torch.matmul(torch.inverse(torch.diag(source_label.sum(axis=0))+torch.eye(self.num_of_class).cuda()),torch.matmul(source_label.T,feature_source_g))
#       self.P=torch.matmul(torch.inverse(torch.diag(source_label.sum(axis=0))),torch.matmul(source_label.T,feature_source_g))
# 分别对原型特征（3*64）和特征（batch_size*64）进行线性变换,分别映射为32*3和32*batch_size
# 原型特征映射后与特征矩阵矩阵映射后乘就是交互特征（batch_size*3）
# 注意到目标域的原型特征使用的是目标域的
       self.stored_mat=torch.matmul(self.V,self.P.T)
       source_predict=torch.matmul(torch.matmul(self.U,feature_source_f.T).T,self.stored_mat)
       target_predict=torch.matmul(torch.matmul(self.U,feature_target_f.T).T,self.stored_mat)
#       source_logit  =source_predict
       source_label_feature=torch.nn.functional.softmax(source_predict, dim=1)
       target_label_feature=torch.nn.functional.softmax(target_predict, dim=1)
       ## DAC part
# 相似矩阵的维度是batch_size*batch_size,利用余弦相似度衡量每两个batch的相似程度
       sim_matrix=self.get_cos_similarity_distance(source_label_feature)
       sim_matrix_target=self.get_cos_similarity_distance(target_label_feature)
       return source_predict,feature_source_f,feature_target_f,sim_matrix,sim_matrix_target
   def compute_target_centroid(self,target,target_label):
    #    因为目标域的原型特征使用的源域的，因此这里是计算目标域的原型特征
       feature_source_g=self.fea_extrator_f(target)
       target_centroid=torch.matmul(torch.inverse(torch.diag(target_label.sum(axis=0))+torch.eye(self.num_of_class).cuda()),torch.matmul(target_label.T,feature_source_g))
       return target_centroid
   def target_domain_evaluation(self,test_features,test_labels):
    #    使用源域的原型特征得到目标域的交互特征，这里是通过softmax得到目标域的标签，计算测试集准确率
       self.eval()
       feature_target_f=self.fea_extrator_f(test_features)
       test_logit=torch.matmul(torch.matmul(self.U,feature_target_f.T).T,self.stored_mat.cuda())
       test_cluster=torch.nn.functional.softmax(test_logit, dim=1)
       test_cluster=np.argmax(test_cluster.cpu().detach().numpy(),axis=1)#.detach()方法，将test_cluster从计算图中取出
       test_labels=np.argmax(test_labels.cpu().detach().numpy(),axis=1)# test_cluster以及test_label从one-hot变成普通标签（0,1,...）
       test_predict=np.zeros_like(test_labels)
       for i in range(len(self.cluster_label)):
           cluster_index=np.where(test_cluster==i)[0]
           test_predict[cluster_index]=self.cluster_label[i]
#       acc=np.sum(label_smooth(test_predict)==test_labels)/len(test_predict)
       acc=np.sum(test_predict==test_labels)/len(test_predict)
       nmi=metrics.normalized_mutual_info_score(test_predict,test_labels)#互信息评价指标
       return acc,nmi   
   def cluster_label_update(self,source_features,source_labels):
       self.eval()
       feature_target_f=self.fea_extrator_f(source_features)
       source_logit=torch.matmul(torch.matmul(self.U,feature_target_f.T).T,self.stored_mat.cuda())#交互特征
       source_cluster=np.argmax(torch.nn.functional.softmax(source_logit, dim=1).cpu().detach().numpy(),axis=1)#将交互特征得到聚类标签
       source_labels=np.argmax(source_labels.cpu().detach().numpy(),axis=1)
       # 将原来的样本聚类，然后让聚类标签和真实标签建立联系
       for i in range(len(self.cluster_label)):
           samples_in_cluster_index=np.where(source_cluster==i)[0]
           label_for_samples=source_labels[samples_in_cluster_index]
           if len(label_for_samples)==0:
              self.cluster_label[i]=0
           else:
              label_for_current_cluster=np.argmax(np.bincount(label_for_samples))
              self.cluster_label[i]=label_for_current_cluster
       source_predict=np.zeros_like(source_labels)
       for i in range(len(self.cluster_label)):
           cluster_index=np.where(source_cluster==i)[0]
           source_predict[cluster_index]=self.cluster_label[i]
       acc=np.sum(source_predict==source_labels)/len(source_predict)
       nmi=metrics.normalized_mutual_info_score(source_predict,source_labels)
       return acc,nmi
   def visualization(self,target,target_labels,tsne=0):
       feature_target_f=self.fea_extrator_f(target)
       target_feature=torch.matmul(torch.matmul(self.U,feature_target_f.T).T,self.stored_mat.cuda())
       #       target_feature=torch.nn.functional.softmax(target_feature, dim=1)
       target_feature=target_feature.cpu().detach().numpy()
       feature_target_f=feature_target_f.cpu().detach().numpy()
       target_labels=np.argmax(target_labels.cpu().detach().numpy(),axis=1)
       colors1 = '#00CED1' #点的颜色
       colors2 = '#DC143C'
       colors3 = '#008000'
       area = np.pi * 4**2  # 点面积 
       if tsne==0:       
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]]
       # 画散点图
           fig = plt.figure()
           ax = Axes3D(fig)
           ax.scatter(x0[:,0],x0[:,1],x0[:,2], s=area, c=colors1, alpha=0.4)
           ax.scatter(x1[:,0],x1[:,1],x1[:,2], s=area, c=colors2, alpha=0.4)
           ax.scatter(x2[:,0],x2[:,1],x2[:,2], s=area, c=colors3, alpha=0.4)
           plt.show()
       else:
           target_feature = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000).fit_transform(feature_target_f.astype('float32'))
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]] 
           plt.scatter(x0[:,0],x0[:,1], s=area, c=colors1, alpha=0.4)
           plt.scatter(x1[:,0],x1[:,1], s=area, c=colors2, alpha=0.4)
           plt.scatter(x2[:,0],x2[:,1], s=area, c=colors3, alpha=0.4)
           plt.show()
   def visualization_4(self,target,target_labels,tsne=0):
       target_feature=self.fea_extrator_f(target)
       #       target_feature=torch.nn.functional.softmax(target_feature, dim=1)
       target_feature=target_feature.cpu().detach().numpy()
       target_labels=np.argmax(target_labels.cpu().detach().numpy(),axis=1)
       colors1 = '#00CED1' #点的颜色
       colors2 = '#DC143C'
       colors3 = '#008000'
       colors4 = '#000000'
       area = np.pi * 4**2  # 点面积 
       if tsne==0:       
           print('error')
           return
       else:
           target_feature = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000).fit_transform(target_feature.astype('float32'))
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]] 
           x3=target_feature[np.where(target_labels==3)[0]] 
           plt.scatter(x0[:,0],x0[:,1], s=area, c=colors1, alpha=0.4)
           plt.scatter(x1[:,0],x1[:,1], s=area, c=colors2, alpha=0.4)
           plt.scatter(x2[:,0],x2[:,1], s=area, c=colors3, alpha=0.4)
           plt.scatter(x3[:,0],x3[:,1], s=area, c=colors4, alpha=0.4)
           plt.show()
   def predict(self,target):
       with torch.no_grad():
           self.eval()         
           feature_target_f=self.fea_extrator_f(target)
           test_logit=torch.matmul(torch.matmul(self.U,feature_target_f.T).T,self.stored_mat.cuda())/8
           test_cluster=torch.nn.functional.softmax(test_logit, dim=1)
           test_cluster=np.argmax(test_cluster.cpu().detach().numpy(),axis=1)
           cluster_0_index,cluster_1_index,cluster_2_index=np.where(test_cluster==0)[0],np.where(test_cluster==1)[0],np.where(test_cluster==2)[0]
           test_cluster[cluster_0_index]=self.cluster_label[0]
           test_cluster[cluster_1_index]=self.cluster_label[1]
           test_cluster[cluster_2_index]=self.cluster_label[2]
       return test_cluster
   def get_cos_similarity_distance(self, features):
        """Get distance in cosine similarity
        :param features: features of samples, (batch_size, num_clusters)
        :return: distance matrix between features, (batch_size, batch_size)
        """
        # (batch_size, num_clusters)
        features_norm = torch.norm(features, dim=1, keepdim=True)#二范数归一化
        # (batch_size, num_clusters)
        features = features / features_norm
        # (batch_size, batch_size)
        cos_dist_matrix = torch.mm(features, features.transpose(0, 1))#计算内积
        return cos_dist_matrix
   def get_cos_similarity_by_threshold(self, cos_dist_matrix):
        """Get similarity by threshold
        :param cos_dist_matrix: cosine distance in matrix,
        (batch_size, batch_size)
        :param threshold: threshold, scalar
        :return: distance matrix between features, (batch_size, batch_size)
        """
        # 用于计算测试集的相似度矩阵
        device = cos_dist_matrix.device
        dtype = cos_dist_matrix.dtype
        similar = torch.tensor(1, dtype=dtype, device=device)
        dissimilar = torch.tensor(0, dtype=dtype, device=device)
        sim_matrix = torch.where(cos_dist_matrix > self.threshold, similar,
                                 dissimilar)
        return sim_matrix
   def compute_indicator(self,cos_dist_matrix):
       device = cos_dist_matrix.device
       dtype = cos_dist_matrix.dtype
       selected = torch.tensor(1, dtype=dtype, device=device)
       not_selected = torch.tensor(0, dtype=dtype, device=device)
       w2=torch.where(cos_dist_matrix < self.lower_threshold,selected,not_selected)
       w1=torch.where(cos_dist_matrix > self.upper_threshold,selected,not_selected)
       w = w1 + w2
       nb_selected=torch.sum(w)
       return w,nb_selected
   def update_threshold(self, epoch: int):
        """Update threshold
        :param threshold: scalar
        :param epoch: scalar
        :return: new_threshold: scalar
        """
        n_epochs = self.max_iter
        diff = self.upper_threshold - self.lower_threshold
        eta = diff / n_epochs
#        eta=self.diff/ n_epochs /2
        # First epoch doesn't update threshold
        if epoch != 0:
            self.upper_threshold = self.upper_threshold-eta
            self.lower_threshold = self.lower_threshold+eta
        else:
            self.upper_threshold = self.upper_threshold
            self.lower_threshold = self.lower_threshold
        self.threshold=(self.upper_threshold+self.lower_threshold)/2
#        print(">>> new threshold is {}".format(new_threshold), flush=True)
   def get_parameters(self) -> List[Dict]:
       params = [
            {"params": self.fea_extrator_f.fc1.parameters(), "lr_mult": 1},
            {"params": self.fea_extrator_f.fc2.parameters(), "lr_mult": 1},
            {"params": self.fea_extrator_g.fc1.parameters(), "lr_mult": 1},
            {"params": self.fea_extrator_g.fc2.parameters(), "lr_mult": 1},
            {"params": self.U, "lr_mult": 1},
            {"params": self.V, "lr_mult": 1},
                ]
       return params  
   
class Domain_adaption_model_withoutproto(nn.Module):
   def __init__(self,hidden_1,hidden_2,hidden_3,hidden_4,num_of_class,low_rank,max_iter,upper_threshold,lower_threshold):
       super(Domain_adaption_model_withoutproto,self).__init__()
       self.fea_extrator_f= feature_extractor(hidden_1,hidden_2)
       self.max_iter=max_iter
       self.upper_threshold=upper_threshold
       self.lower_threshold=lower_threshold
       self.classifier=nn.Linear(hidden_2,num_of_class)
#       self.diff=(upper_threshold-lower_threshold)
       self.threshold=upper_threshold
       self.cluster_label=np.zeros(num_of_class)
       self.num_of_class=num_of_class
   def forward(self,source,target,source_label):
       feature_source_f=self.fea_extrator_f(source)
       feature_target_f=self.fea_extrator_f(target)
       ##torch.matmul(source_label.T,torch.ones(batch_num,num_of_class))
       ## Update P through some algebra computations for the convenice of broadcast
       source_predict=self.classifier(feature_source_f)
       target_predict=self.classifier(feature_target_f)
#       source_logit  =source_predict
       source_label_feature=torch.nn.functional.softmax(source_predict, dim=1)
       target_label_feature=torch.nn.functional.softmax(target_predict, dim=1)
       ## DAC part
       sim_matrix=self.get_cos_similarity_distance(source_label_feature)
       sim_matrix_target=self.get_cos_similarity_distance(target_label_feature)
       return source_predict,feature_source_f,feature_target_f,sim_matrix,sim_matrix_target
   def compute_target_centroid(self,target,target_label):
       feature_source_g=self.fea_extrator_f(target)
       target_centroid=torch.matmul(torch.inverse(torch.diag(target_label.sum(axis=0))+torch.eye(self.num_of_class).cuda()),torch.matmul(target_label.T,feature_source_g))
       return target_centroid
   def target_domain_evaluation(self,test_features,test_labels):
       self.eval()
       feature_target_f=self.fea_extrator_f(test_features)
       test_logit=self.classifier(feature_target_f)
       test_cluster=torch.nn.functional.softmax(test_logit, dim=1)
       test_cluster=np.argmax(test_cluster.cpu().detach().numpy(),axis=1)
       test_labels=np.argmax(test_labels.cpu().detach().numpy(),axis=1)
       test_predict=np.zeros_like(test_labels)
       for i in range(len(self.cluster_label)):
           cluster_index=np.where(test_cluster==i)[0]
           test_predict[cluster_index]=self.cluster_label[i]
#       acc=np.sum(label_smooth(test_predict)==test_labels)/len(test_predict)
       acc=np.sum(test_predict==test_labels)/len(test_predict)
       nmi=metrics.normalized_mutual_info_score(test_predict,test_labels)
       return acc,nmi   
   def cluster_label_update(self,source_features,source_labels):
       self.eval()
       feature_target_f=self.fea_extrator_f(source_features)
       source_logit=self.classifier(feature_target_f)
       source_cluster=np.argmax(torch.nn.functional.softmax(source_logit, dim=1).cpu().detach().numpy(),axis=1)
       source_labels=np.argmax(source_labels.cpu().detach().numpy(),axis=1)
       for i in range(len(self.cluster_label)):
           samples_in_cluster_index=np.where(source_cluster==i)[0]
           label_for_samples=source_labels[samples_in_cluster_index]
           if len(label_for_samples)==0:
              self.cluster_label[i]=0
           else:
              label_for_current_cluster=np.argmax(np.bincount(label_for_samples))
              self.cluster_label[i]=label_for_current_cluster
       source_predict=np.zeros_like(source_labels)
       for i in range(len(self.cluster_label)):
           cluster_index=np.where(source_cluster==i)[0]
           source_predict[cluster_index]=self.cluster_label[i]
       acc=np.sum(source_predict==source_labels)/len(source_predict)
       nmi=metrics.normalized_mutual_info_score(source_predict,source_labels)
       return acc,nmi
   def visualization(self,target,target_labels,tsne=0):
       feature_target_f=self.fea_extrator_f(target)
       target_feature=self.classifier(feature_target_f)
       #       target_feature=torch.nn.functional.softmax(target_feature, dim=1)
       target_feature=target_feature.cpu().detach().numpy()
       feature_target_f=feature_target_f.cpu().detach().numpy()
       target_labels=np.argmax(target_labels.cpu().detach().numpy(),axis=1)
       colors1 = '#00CED1' #点的颜色
       colors2 = '#DC143C'
       colors3 = '#008000'
       area = np.pi * 4**2  # 点面积 
       if tsne==0:       
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]]
       # 画散点图
           fig = plt.figure()
           ax = Axes3D(fig)
           ax.scatter(x0[:,0],x0[:,1],x0[:,2], s=area, c=colors1, alpha=0.4)
           ax.scatter(x1[:,0],x1[:,1],x1[:,2], s=area, c=colors2, alpha=0.4)
           ax.scatter(x2[:,0],x2[:,1],x2[:,2], s=area, c=colors3, alpha=0.4)
           plt.show()
       else:
           target_feature = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000).fit_transform(feature_target_f.astype('float32'))
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]] 
           plt.scatter(x0[:,0],x0[:,1], s=area, c=colors1, alpha=0.4)
           plt.scatter(x1[:,0],x1[:,1], s=area, c=colors2, alpha=0.4)
           plt.scatter(x2[:,0],x2[:,1], s=area, c=colors3, alpha=0.4)
           plt.show()
   def visualization_4(self,target,target_labels,tsne=0):
       target_feature=self.fea_extrator_f(target)
       #       target_feature=torch.nn.functional.softmax(target_feature, dim=1)
       target_feature=target_feature.cpu().detach().numpy()
       target_labels=np.argmax(target_labels.cpu().detach().numpy(),axis=1)
       colors1 = '#00CED1' #点的颜色
       colors2 = '#DC143C'
       colors3 = '#008000'
       colors4 = '#000000'
       area = np.pi * 4**2  # 点面积 
       if tsne==0:       
           print('error')
           return
       else:
           target_feature = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000).fit_transform(target_feature.astype('float32'))
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]] 
           x3=target_feature[np.where(target_labels==3)[0]] 
           plt.scatter(x0[:,0],x0[:,1], s=area, c=colors1, alpha=0.4)
           plt.scatter(x1[:,0],x1[:,1], s=area, c=colors2, alpha=0.4)
           plt.scatter(x2[:,0],x2[:,1], s=area, c=colors3, alpha=0.4)
           plt.scatter(x3[:,0],x3[:,1], s=area, c=colors4, alpha=0.4)
           plt.show()
   def get_cos_similarity_distance(self, features):
        """Get distance in cosine similarity
        :param features: features of samples, (batch_size, num_clusters)
        :return: distance matrix between features, (batch_size, batch_size)
        """
        # (batch_size, num_clusters)
        features_norm = torch.norm(features, dim=1, keepdim=True)
        # (batch_size, num_clusters)
        features = features / features_norm
        # (batch_size, batch_size)
        cos_dist_matrix = torch.mm(features, features.transpose(0, 1))
        return cos_dist_matrix
   def get_cos_similarity_by_threshold(self, cos_dist_matrix):
        """Get similarity by threshold
        :param cos_dist_matrix: cosine distance in matrix,
        (batch_size, batch_size)
        :param threshold: threshold, scalar
        :return: distance matrix between features, (batch_size, batch_size)
        """
        # 测试集的相似度矩阵由源域的相似度矩阵取阈值二值化确定
        device = cos_dist_matrix.device
        dtype = cos_dist_matrix.dtype
        similar = torch.tensor(1, dtype=dtype, device=device)
        dissimilar = torch.tensor(0, dtype=dtype, device=device)
        sim_matrix = torch.where(cos_dist_matrix > self.threshold, similar,
                                 dissimilar)
        return sim_matrix
   def compute_indicator(self,cos_dist_matrix):
       device = cos_dist_matrix.device
       dtype = cos_dist_matrix.dtype
       selected = torch.tensor(1, dtype=dtype, device=device)
       not_selected = torch.tensor(0, dtype=dtype, device=device)
       w2=torch.where(cos_dist_matrix < self.lower_threshold,selected,not_selected)
       w1=torch.where(cos_dist_matrix > self.upper_threshold,selected,not_selected)
       w = w1 + w2
       nb_selected=torch.sum(w)
       return w,nb_selected
   def update_threshold(self, epoch: int):
        """Update threshold
        :param threshold: scalar
        :param epoch: scalar
        :return: new_threshold: scalar
        """
        n_epochs = self.max_iter
        diff = self.upper_threshold - self.lower_threshold
        eta = diff / n_epochs
#        eta=self.diff/ n_epochs /2
        # First epoch doesn't update threshold
        if epoch != 0:
            self.upper_threshold = self.upper_threshold-eta
            self.lower_threshold = self.lower_threshold+eta
        else:
            self.upper_threshold = self.upper_threshold
            self.lower_threshold = self.lower_threshold
        self.threshold=(self.upper_threshold+self.lower_threshold)/2
#        print(">>> new threshold is {}".format(new_threshold), flush=True)
   def get_parameters(self) -> List[Dict]:
       params = [
            {"params": self.fea_extrator_f.fc1.parameters(), "lr_mult": 1},
            {"params": self.fea_extrator_f.fc2.parameters(), "lr_mult": 1},
            {"params": self.classifier.parameters(), "lr_mult": 1},
                ]
       return params
   
class Domain_adaption_model_withoutproto_withoutpair(nn.Module):
   def __init__(self,hidden_1,hidden_2,hidden_3,hidden_4,num_of_class,low_rank,max_iter,upper_threshold,lower_threshold):
       super(Domain_adaption_model_withoutproto_withoutpair,self).__init__()
       self.fea_extrator_f= feature_extractor(hidden_1,hidden_2)
       self.max_iter=max_iter
    #    self.upper_threshold=upper_threshold
    #    self.lower_threshold=lower_threshold
       self.classifier=nn.Linear(hidden_2,num_of_class)
#       self.diff=(upper_threshold-lower_threshold)
    #    self.threshold=upper_threshold
    #    self.cluster_label=np.zeros(num_of_class)
    #    self.num_of_class=num_of_class
   def forward(self,source,target,source_label):
       feature_source_f=self.fea_extrator_f(source)
       feature_target_f=self.fea_extrator_f(target)
       ##torch.matmul(source_label.T,torch.ones(batch_num,num_of_class))
       ## Update P through some algebra computations for the convenice of broadcast
       source_predict=self.classifier(feature_source_f)
       target_predict=self.classifier(feature_target_f)
#       source_logit  =source_predict
       source_label_feature=torch.nn.functional.softmax(source_predict, dim=1)
       target_label_feature=torch.nn.functional.softmax(target_predict, dim=1)
    #    ## DAC part
    #    sim_matrix=self.get_cos_similarity_distance(source_label_feature)
    #    sim_matrix_target=self.get_cos_similarity_distance(target_label_feature)
       return source_predict,feature_source_f,feature_target_f,source_label_feature,target_label_feature
#    def compute_target_centroid(self,target,target_label):
#        feature_source_g=self.fea_extrator_f(target)
#        target_centroid=torch.matmul(torch.inverse(torch.diag(target_label.sum(axis=0))+torch.eye(self.num_of_class).cuda()),torch.matmul(target_label.T,feature_source_g))
#        return target_centroid
   def target_domain_evaluation(self,test_features,test_labels):
       self.eval()
       feature_target_f=self.fea_extrator_f(test_features)
       test_logit=self.classifier(feature_target_f)
       test_cluster=torch.nn.functional.softmax(test_logit, dim=1)
       test_cluster=np.argmax(test_cluster.cpu().detach().numpy(),axis=1)
       test_labels=np.argmax(test_labels.cpu().detach().numpy(),axis=1)

       test_predict = test_cluster

    #    test_predict=np.zeros_like(test_labels)
    #    for i in range(len(self.cluster_label)):
    #        cluster_index=np.where(test_cluster==i)[0]
    #        test_predict[cluster_index]=self.cluster_label[i]
#       acc=np.sum(label_smooth(test_predict)==test_labels)/len(test_predict)
       acc=np.sum(test_predict==test_labels)/len(test_predict)
       nmi=metrics.normalized_mutual_info_score(test_predict,test_labels)
       return acc,nmi   
   def cluster_label_update(self,source_features,source_labels):
       self.eval()
       feature_target_f=self.fea_extrator_f(source_features)
       source_logit=self.classifier(feature_target_f)
       source_cluster=np.argmax(torch.nn.functional.softmax(source_logit, dim=1).cpu().detach().numpy(),axis=1)
       source_labels=np.argmax(source_labels.cpu().detach().numpy(),axis=1)
    #    for i in range(len(self.cluster_label)):
    #        samples_in_cluster_index=np.where(source_cluster==i)[0]
    #        label_for_samples=source_labels[samples_in_cluster_index]
    #        if len(label_for_samples)==0:
    #           self.cluster_label[i]=0
    #        else:
    #           label_for_current_cluster=np.argmax(np.bincount(label_for_samples))
    #           self.cluster_label[i]=label_for_current_cluster
    #    source_predict=np.zeros_like(source_labels)
    #    for i in range(len(self.cluster_label)):
    #        cluster_index=np.where(source_cluster==i)[0]
    #        source_predict[cluster_index]=self.cluster_label[i]


       source_predict = source_cluster

       acc=np.sum(source_predict==source_labels)/len(source_predict)
       nmi=metrics.normalized_mutual_info_score(source_predict,source_labels)
       return acc,nmi
   def visualization(self,target,target_labels,tsne=0):
       feature_target_f=self.fea_extrator_f(target)
       target_feature=self.classifier(feature_target_f)
       #       target_feature=torch.nn.functional.softmax(target_feature, dim=1)
       target_feature=target_feature.cpu().detach().numpy()
       feature_target_f=feature_target_f.cpu().detach().numpy()
       target_labels=np.argmax(target_labels.cpu().detach().numpy(),axis=1)
       colors1 = '#00CED1' #点的颜色
       colors2 = '#DC143C'
       colors3 = '#008000'
       area = np.pi * 4**2  # 点面积 
       if tsne==0:       
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]]
       # 画散点图
           fig = plt.figure()
           ax = Axes3D(fig)
           ax.scatter(x0[:,0],x0[:,1],x0[:,2], s=area, c=colors1, alpha=0.4)
           ax.scatter(x1[:,0],x1[:,1],x1[:,2], s=area, c=colors2, alpha=0.4)
           ax.scatter(x2[:,0],x2[:,1],x2[:,2], s=area, c=colors3, alpha=0.4)
           plt.show()
       else:
           target_feature = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000).fit_transform(feature_target_f.astype('float32'))
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]] 
           plt.scatter(x0[:,0],x0[:,1], s=area, c=colors1, alpha=0.4)
           plt.scatter(x1[:,0],x1[:,1], s=area, c=colors2, alpha=0.4)
           plt.scatter(x2[:,0],x2[:,1], s=area, c=colors3, alpha=0.4)
           plt.show()
   def visualization_4(self,target,target_labels,tsne=0):
       target_feature=self.fea_extrator_f(target)
       #       target_feature=torch.nn.functional.softmax(target_feature, dim=1)
       target_feature=target_feature.cpu().detach().numpy()
       target_labels=np.argmax(target_labels.cpu().detach().numpy(),axis=1)
       colors1 = '#00CED1' #点的颜色
       colors2 = '#DC143C'
       colors3 = '#008000'
       colors4 = '#000000'
       area = np.pi * 4**2  # 点面积 
       if tsne==0:       
           print('error')
           return
       else:
           target_feature = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000).fit_transform(target_feature.astype('float32'))
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]] 
           x3=target_feature[np.where(target_labels==3)[0]] 
           plt.scatter(x0[:,0],x0[:,1], s=area, c=colors1, alpha=0.4)
           plt.scatter(x1[:,0],x1[:,1], s=area, c=colors2, alpha=0.4)
           plt.scatter(x2[:,0],x2[:,1], s=area, c=colors3, alpha=0.4)
           plt.scatter(x3[:,0],x3[:,1], s=area, c=colors4, alpha=0.4)
           plt.show()
#    def get_cos_similarity_distance(self, features):
#         """Get distance in cosine similarity
#         :param features: features of samples, (batch_size, num_clusters)
#         :return: distance matrix between features, (batch_size, batch_size)
#         """
#         # (batch_size, num_clusters)
#         features_norm = torch.norm(features, dim=1, keepdim=True)
#         # (batch_size, num_clusters)
#         features = features / features_norm
#         # (batch_size, batch_size)
#         cos_dist_matrix = torch.mm(features, features.transpose(0, 1))
#         return cos_dist_matrix
#    def get_cos_similarity_by_threshold(self, cos_dist_matrix):
#         """Get similarity by threshold
#         :param cos_dist_matrix: cosine distance in matrix,
#         (batch_size, batch_size)
#         :param threshold: threshold, scalar
#         :return: distance matrix between features, (batch_size, batch_size)
#         """
#         # 测试集的相似度矩阵由源域的相似度矩阵取阈值二值化确定
#         device = cos_dist_matrix.device
#         dtype = cos_dist_matrix.dtype
#         similar = torch.tensor(1, dtype=dtype, device=device)
#         dissimilar = torch.tensor(0, dtype=dtype, device=device)
#         sim_matrix = torch.where(cos_dist_matrix > self.threshold, similar,
#                                  dissimilar)
#         return sim_matrix
#    def compute_indicator(self,cos_dist_matrix):
#        device = cos_dist_matrix.device
#        dtype = cos_dist_matrix.dtype
#        selected = torch.tensor(1, dtype=dtype, device=device)
#        not_selected = torch.tensor(0, dtype=dtype, device=device)
#        w2=torch.where(cos_dist_matrix < self.lower_threshold,selected,not_selected)
#        w1=torch.where(cos_dist_matrix > self.upper_threshold,selected,not_selected)
#        w = w1 + w2
#        nb_selected=torch.sum(w)
#        return w,nb_selected
#    def update_threshold(self, epoch: int):
#         """Update threshold
#         :param threshold: scalar
#         :param epoch: scalar
#         :return: new_threshold: scalar
#         """
#         n_epochs = self.max_iter
#         diff = self.upper_threshold - self.lower_threshold
#         eta = diff / n_epochs
# #        eta=self.diff/ n_epochs /2
#         # First epoch doesn't update threshold
#         if epoch != 0:
#             self.upper_threshold = self.upper_threshold-eta
#             self.lower_threshold = self.lower_threshold+eta
#         else:
#             self.upper_threshold = self.upper_threshold
#             self.lower_threshold = self.lower_threshold
#         self.threshold=(self.upper_threshold+self.lower_threshold)/2
# #        print(">>> new threshold is {}".format(new_threshold), flush=True)
   def get_parameters(self) -> List[Dict]:
       params = [
            {"params": self.fea_extrator_f.fc1.parameters(), "lr_mult": 1},
            {"params": self.fea_extrator_f.fc2.parameters(), "lr_mult": 1},
            {"params": self.classifier.parameters(), "lr_mult": 1},
                ]
       return params


class feature_extractor_LSTM(nn.Module):
    def __init__(self,hidden_1,hidden_2, input_dim=5, hidden_dim=32, num_layers=2):
         super(feature_extractor_LSTM,self).__init__()
         self.lstm=nn.LSTM(input_dim, hidden_dim,num_layers,batch_first=True)
         self.fc1=nn.Linear(62*hidden_dim,hidden_1)
         self.fc2=nn.Linear(hidden_1,hidden_2)
         self.dropout1 = nn.Dropout(p=0.25)
         self.dropout2 = nn.Dropout(p=0.25)

         self.hidden_dim = hidden_dim

    def forward(self,x):
         x,_=self.lstm(x)
         x=x.reshape(-1, 62*self.hidden_dim)
         x=self.fc1(x)
         x=F.relu(x)
#         x=F.leaky_relu(x)
         x=self.fc2(x)
         x=F.relu(x)
#         x=F.leaky_relu(x)
         return x
    def get_parameters(self) -> List[Dict]:
         params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
                  ]
         return params  
    
# class classifier_transformer(nn.Module):
#     def __init__(self,hidden_2,num_of_class,hidden_dim=32,num_heads=2,num_layers=2,dropout=0.5):
#         super(classifier_transformer).__init__()

#         # hidden_dim = 32
#         # num_heads = 2
#         # num_layers = 2
#         # dropout = 0.5

#         self.transformer = nn.Transformer(nhead=num_heads, \
#                                           num_encoder_layers=num_layers,
#                                           num_decoder_layers=num_layers, dropout=dropout,batch_first=True)


#         self.Linear = nn.Linear(hidden_2,num_of_class)

#         self.hidden_dim = hidden_dim

#     def forward(self,x):

#         x = self.transformer(x)
#         x = x.reshape(-1, self.hidden_dim)
#         x = self.Linear(x)

#         return x


class Dann_withLSTM(nn.Module):
   def __init__(self,hidden_1,hidden_2,num_of_class,max_iter,lstm_input_dim=5, lstm_hidden_dim=32, lstm_num_layers=2):
       super(Dann_withLSTM,self).__init__()
       self.fea_extrator_f= feature_extractor_LSTM(hidden_1,hidden_2,lstm_input_dim,lstm_hidden_dim,lstm_num_layers)
       self.max_iter=max_iter
    #    self.upper_threshold=upper_threshold
    #    self.lower_threshold=lower_threshold
       self.classifier=nn.Linear(hidden_2,num_of_class)
    #    self.classifier1=classifier_transformer(hidden_2,num_of_class)
#       self.diff=(upper_threshold-lower_threshold)
    #    self.threshold=upper_threshold
    #    self.cluster_label=np.zeros(num_of_class)
    #    self.num_of_class=num_of_class
   def forward(self,source,target,source_label):
       feature_source_f=self.fea_extrator_f(source)
       feature_target_f=self.fea_extrator_f(target)
       ##torch.matmul(source_label.T,torch.ones(batch_num,num_of_class))
       ## Update P through some algebra computations for the convenice of broadcast
       source_predict=self.classifier(feature_source_f)
       target_predict=self.classifier(feature_target_f)
#       source_logit  =source_predict
       source_label_feature=torch.nn.functional.softmax(source_predict, dim=1)
       target_label_feature=torch.nn.functional.softmax(target_predict, dim=1)
    #    ## DAC part
    #    sim_matrix=self.get_cos_similarity_distance(source_label_feature)
    #    sim_matrix_target=self.get_cos_similarity_distance(target_label_feature)
       return source_predict,feature_source_f,feature_target_f,source_label_feature,target_label_feature
#    def compute_target_centroid(self,target,target_label):
#        feature_source_g=self.fea_extrator_f(target)
#        target_centroid=torch.matmul(torch.inverse(torch.diag(target_label.sum(axis=0))+torch.eye(self.num_of_class).cuda()),torch.matmul(target_label.T,feature_source_g))
#        return target_centroid
   def target_domain_evaluation(self,test_features,test_labels):
       self.eval()
       feature_target_f=self.fea_extrator_f(test_features)
       test_logit=self.classifier(feature_target_f)
       test_cluster=torch.nn.functional.softmax(test_logit, dim=1)
       test_cluster=np.argmax(test_cluster.cpu().detach().numpy(),axis=1)
       test_labels=np.argmax(test_labels.cpu().detach().numpy(),axis=1)

       test_predict = test_cluster

    #    test_predict=np.zeros_like(test_labels)
    #    for i in range(len(self.cluster_label)):
    #        cluster_index=np.where(test_cluster==i)[0]
    #        test_predict[cluster_index]=self.cluster_label[i]
#       acc=np.sum(label_smooth(test_predict)==test_labels)/len(test_predict)
       acc=np.sum(test_predict==test_labels)/len(test_predict)
       nmi=metrics.normalized_mutual_info_score(test_predict,test_labels)
       return acc,nmi   
   def cluster_label_update(self,source_features,source_labels):
       self.eval()
       feature_target_f=self.fea_extrator_f(source_features)
       source_logit=self.classifier(feature_target_f)
       source_cluster=np.argmax(torch.nn.functional.softmax(source_logit, dim=1).cpu().detach().numpy(),axis=1)
       source_labels=np.argmax(source_labels.cpu().detach().numpy(),axis=1)
    #    for i in range(len(self.cluster_label)):
    #        samples_in_cluster_index=np.where(source_cluster==i)[0]
    #        label_for_samples=source_labels[samples_in_cluster_index]
    #        if len(label_for_samples)==0:
    #           self.cluster_label[i]=0
    #        else:
    #           label_for_current_cluster=np.argmax(np.bincount(label_for_samples))
    #           self.cluster_label[i]=label_for_current_cluster
    #    source_predict=np.zeros_like(source_labels)
    #    for i in range(len(self.cluster_label)):
    #        cluster_index=np.where(source_cluster==i)[0]
    #        source_predict[cluster_index]=self.cluster_label[i]


       source_predict = source_cluster

       acc=np.sum(source_predict==source_labels)/len(source_predict)
       nmi=metrics.normalized_mutual_info_score(source_predict,source_labels)
       return acc,nmi
   def visualization(self,target,target_labels,tsne=0):
       feature_target_f=self.fea_extrator_f(target)
       target_feature=self.classifier(feature_target_f)
       #       target_feature=torch.nn.functional.softmax(target_feature, dim=1)
       target_feature=target_feature.cpu().detach().numpy()
       feature_target_f=feature_target_f.cpu().detach().numpy()
       target_labels=np.argmax(target_labels.cpu().detach().numpy(),axis=1)
       colors1 = '#00CED1' #点的颜色
       colors2 = '#DC143C'
       colors3 = '#008000'
       area = np.pi * 4**2  # 点面积 
       if tsne==0:       
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]]
       # 画散点图
           fig = plt.figure()
           ax = Axes3D(fig)
           ax.scatter(x0[:,0],x0[:,1],x0[:,2], s=area, c=colors1, alpha=0.4)
           ax.scatter(x1[:,0],x1[:,1],x1[:,2], s=area, c=colors2, alpha=0.4)
           ax.scatter(x2[:,0],x2[:,1],x2[:,2], s=area, c=colors3, alpha=0.4)
           plt.show()
       else:
           target_feature = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000).fit_transform(feature_target_f.astype('float32'))
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]] 
           plt.scatter(x0[:,0],x0[:,1], s=area, c=colors1, alpha=0.4)
           plt.scatter(x1[:,0],x1[:,1], s=area, c=colors2, alpha=0.4)
           plt.scatter(x2[:,0],x2[:,1], s=area, c=colors3, alpha=0.4)
           plt.show()
   def visualization_4(self,target,target_labels,tsne=0):
       target_feature=self.fea_extrator_f(target)
       #       target_feature=torch.nn.functional.softmax(target_feature, dim=1)
       target_feature=target_feature.cpu().detach().numpy()
       target_labels=np.argmax(target_labels.cpu().detach().numpy(),axis=1)
       colors1 = '#00CED1' #点的颜色
       colors2 = '#DC143C'
       colors3 = '#008000'
       colors4 = '#000000'
       area = np.pi * 4**2  # 点面积 
       if tsne==0:       
           print('error')
           return
       else:
           target_feature = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000).fit_transform(target_feature.astype('float32'))
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]] 
           x3=target_feature[np.where(target_labels==3)[0]] 
           plt.scatter(x0[:,0],x0[:,1], s=area, c=colors1, alpha=0.4)
           plt.scatter(x1[:,0],x1[:,1], s=area, c=colors2, alpha=0.4)
           plt.scatter(x2[:,0],x2[:,1], s=area, c=colors3, alpha=0.4)
           plt.scatter(x3[:,0],x3[:,1], s=area, c=colors4, alpha=0.4)
           plt.show()
#    def get_cos_similarity_distance(self, features):
#         """Get distance in cosine similarity
#         :param features: features of samples, (batch_size, num_clusters)
#         :return: distance matrix between features, (batch_size, batch_size)
#         """
#         # (batch_size, num_clusters)
#         features_norm = torch.norm(features, dim=1, keepdim=True)
#         # (batch_size, num_clusters)
#         features = features / features_norm
#         # (batch_size, batch_size)
#         cos_dist_matrix = torch.mm(features, features.transpose(0, 1))
#         return cos_dist_matrix
#    def get_cos_similarity_by_threshold(self, cos_dist_matrix):
#         """Get similarity by threshold
#         :param cos_dist_matrix: cosine distance in matrix,
#         (batch_size, batch_size)
#         :param threshold: threshold, scalar
#         :return: distance matrix between features, (batch_size, batch_size)
#         """
#         # 测试集的相似度矩阵由源域的相似度矩阵取阈值二值化确定
#         device = cos_dist_matrix.device
#         dtype = cos_dist_matrix.dtype
#         similar = torch.tensor(1, dtype=dtype, device=device)
#         dissimilar = torch.tensor(0, dtype=dtype, device=device)
#         sim_matrix = torch.where(cos_dist_matrix > self.threshold, similar,
#                                  dissimilar)
#         return sim_matrix
#    def compute_indicator(self,cos_dist_matrix):
#        device = cos_dist_matrix.device
#        dtype = cos_dist_matrix.dtype
#        selected = torch.tensor(1, dtype=dtype, device=device)
#        not_selected = torch.tensor(0, dtype=dtype, device=device)
#        w2=torch.where(cos_dist_matrix < self.lower_threshold,selected,not_selected)
#        w1=torch.where(cos_dist_matrix > self.upper_threshold,selected,not_selected)
#        w = w1 + w2
#        nb_selected=torch.sum(w)
#        return w,nb_selected
#    def update_threshold(self, epoch: int):
#         """Update threshold
#         :param threshold: scalar
#         :param epoch: scalar
#         :return: new_threshold: scalar
#         """
#         n_epochs = self.max_iter
#         diff = self.upper_threshold - self.lower_threshold
#         eta = diff / n_epochs
# #        eta=self.diff/ n_epochs /2
#         # First epoch doesn't update threshold
#         if epoch != 0:
#             self.upper_threshold = self.upper_threshold-eta
#             self.lower_threshold = self.lower_threshold+eta
#         else:
#             self.upper_threshold = self.upper_threshold
#             self.lower_threshold = self.lower_threshold
#         self.threshold=(self.upper_threshold+self.lower_threshold)/2
# #        print(">>> new threshold is {}".format(new_threshold), flush=True)
   def get_parameters(self) -> List[Dict]:
       params = [
            {"params": self.fea_extrator_f.fc1.parameters(), "lr_mult": 1},
            {"params": self.fea_extrator_f.fc2.parameters(), "lr_mult": 1},
            {"params": self.classifier.parameters(), "lr_mult": 1},
                ]
       return params
   


# ExternalAttention机制
class ExternalAttention(nn.Module):

  def __init__(self, d_model, S=62):
    super().__init__()
    self.mk=nn.Linear(d_model,S,bias=False)
    self.mv=nn.Linear(S,d_model,bias=False)
    self.softmax=nn.Softmax(dim=1)
    self.init_weights()


  def init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)

  def forward(self, queries):
    attn=self.mk(queries) # bs, n, S
    attn=self.softmax(attn) # bs, n, S
    attn=attn/torch.sum(attn,dim=2,keepdim=True) # bs,n,S
    out=self.mv(attn) # bs, n, d_model

    return out
  

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.randn(hidden_size, 1))

    def forward(self, input_tensor):
        # 计算注意力权重
        batch_size = input_tensor.size(0)
        seq_len = input_tensor.size(1)
        att_scores = torch.matmul(input_tensor, self.att_weights)
        att_scores = torch.tanh(att_scores)
        att_scores = torch.exp(att_scores)
        att_scores = att_scores / torch.sum(att_scores, dim=1, keepdim=True)

        # 加权求和
        att_output = torch.bmm(input_tensor.transpose(1,2), att_scores)
        att_output = att_output.view(batch_size, self.hidden_size)

        return att_output

class feature_extractor_LSTM_att(nn.Module):
    def __init__(self,hidden_1,hidden_2, input_dim=5, hidden_dim=32, num_layers=2):
         super(feature_extractor_LSTM_att,self).__init__()
         self.lstm=nn.LSTM(input_dim, hidden_dim,num_layers,batch_first=True)
         self.fc1=nn.Linear(62*hidden_dim,hidden_1)
         self.fc2=nn.Linear(hidden_1,hidden_2)
         self.dropout1 = nn.Dropout(p=0.25)
         self.dropout2 = nn.Dropout(p=0.25)

         hidden_size = hidden_dim
         self.W_w = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
         self.u_w = nn.Parameter(torch.Tensor(hidden_size,1))

         nn.init.uniform_(self.W_w,-0.1,0.1)
         nn.init.uniform_(self.u_w,-0.1,0.1)

    def forward(self,x):
         x,_=self.lstm(x)# batch_size*seq_len*emd_dim
         
         # attention
         score = torch.tanh(torch.matmul(x, self.W_w))# batch_size*seq_len*emd_dim
         attention_weights = F.softmax(torch.matmul(score, self.u_w),dim=1)

         score_x = x * attention_weights

         #
         score_x = score_x.reshape(-1, 62*32)
         score_x = self.fc1(score_x)
         score_x = F.relu(score_x)
         score_x = self.fc2(score_x)

         score_x = F.relu(score_x)

         return score_x


class Dann_withLSTM_att(nn.Module):
   def __init__(self,hidden_1,hidden_2,num_of_class,max_iter,lstm_input_dim=5, lstm_hidden_dim=32, lstm_num_layers=2):
       super(Dann_withLSTM_att,self).__init__()
       self.fea_extrator_f= feature_extractor_LSTM(hidden_1,hidden_2,lstm_input_dim,lstm_hidden_dim,lstm_num_layers)
       self.max_iter=max_iter
    #    self.upper_threshold=upper_threshold
    #    self.lower_threshold=lower_threshold
       self.classifier=nn.Linear(hidden_2,num_of_class)
    #    self.classifier1=classifier_transformer(hidden_2,num_of_class)
#       self.diff=(upper_threshold-lower_threshold)
    #    self.threshold=upper_threshold
    #    self.cluster_label=np.zeros(num_of_class)
    #    self.num_of_class=num_of_class
   def forward(self,source,target,source_label):
       feature_source_f=self.fea_extrator_f(source)
       feature_target_f=self.fea_extrator_f(target)
       ##torch.matmul(source_label.T,torch.ones(batch_num,num_of_class))
       ## Update P through some algebra computations for the convenice of broadcast
       source_predict=self.classifier(feature_source_f)
       target_predict=self.classifier(feature_target_f)
#       source_logit  =source_predict
       source_label_feature=torch.nn.functional.softmax(source_predict, dim=1)
       target_label_feature=torch.nn.functional.softmax(target_predict, dim=1)
    #    ## DAC part
    #    sim_matrix=self.get_cos_similarity_distance(source_label_feature)
    #    sim_matrix_target=self.get_cos_similarity_distance(target_label_feature)
       return source_predict,feature_source_f,feature_target_f,source_label_feature,target_label_feature
#    def compute_target_centroid(self,target,target_label):
#        feature_source_g=self.fea_extrator_f(target)
#        target_centroid=torch.matmul(torch.inverse(torch.diag(target_label.sum(axis=0))+torch.eye(self.num_of_class).cuda()),torch.matmul(target_label.T,feature_source_g))
#        return target_centroid
   def target_domain_evaluation(self,test_features,test_labels):
       self.eval()
       feature_target_f=self.fea_extrator_f(test_features)
       test_logit=self.classifier(feature_target_f)
       test_cluster=torch.nn.functional.softmax(test_logit, dim=1)
       test_cluster=np.argmax(test_cluster.cpu().detach().numpy(),axis=1)
       test_labels=np.argmax(test_labels.cpu().detach().numpy(),axis=1)

       test_predict = test_cluster

    #    test_predict=np.zeros_like(test_labels)
    #    for i in range(len(self.cluster_label)):
    #        cluster_index=np.where(test_cluster==i)[0]
    #        test_predict[cluster_index]=self.cluster_label[i]
#       acc=np.sum(label_smooth(test_predict)==test_labels)/len(test_predict)
       acc=np.sum(test_predict==test_labels)/len(test_predict)
       nmi=metrics.normalized_mutual_info_score(test_predict,test_labels)
       return acc,nmi   
   def cluster_label_update(self,source_features,source_labels):
       self.eval()
       feature_target_f=self.fea_extrator_f(source_features)
       source_logit=self.classifier(feature_target_f)
       source_cluster=np.argmax(torch.nn.functional.softmax(source_logit, dim=1).cpu().detach().numpy(),axis=1)
       source_labels=np.argmax(source_labels.cpu().detach().numpy(),axis=1)
    


       source_predict = source_cluster

       acc=np.sum(source_predict==source_labels)/len(source_predict)
       nmi=metrics.normalized_mutual_info_score(source_predict,source_labels)
       return acc,nmi
   def visualization(self,target,target_labels,tsne=0):
       feature_target_f=self.fea_extrator_f(target)
       target_feature=self.classifier(feature_target_f)
       #       target_feature=torch.nn.functional.softmax(target_feature, dim=1)
       target_feature=target_feature.cpu().detach().numpy()
       feature_target_f=feature_target_f.cpu().detach().numpy()
       target_labels=np.argmax(target_labels.cpu().detach().numpy(),axis=1)
       colors1 = '#00CED1' #点的颜色
       colors2 = '#DC143C'
       colors3 = '#008000'
       area = np.pi * 4**2  # 点面积 
       if tsne==0:       
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]]
       # 画散点图
           fig = plt.figure()
           ax = Axes3D(fig)
           ax.scatter(x0[:,0],x0[:,1],x0[:,2], s=area, c=colors1, alpha=0.4)
           ax.scatter(x1[:,0],x1[:,1],x1[:,2], s=area, c=colors2, alpha=0.4)
           ax.scatter(x2[:,0],x2[:,1],x2[:,2], s=area, c=colors3, alpha=0.4)
           plt.show()
       else:
           target_feature = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000).fit_transform(feature_target_f.astype('float32'))
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]] 
           plt.scatter(x0[:,0],x0[:,1], s=area, c=colors1, alpha=0.4)
           plt.scatter(x1[:,0],x1[:,1], s=area, c=colors2, alpha=0.4)
           plt.scatter(x2[:,0],x2[:,1], s=area, c=colors3, alpha=0.4)
           plt.show()
   def visualization_4(self,target,target_labels,tsne=0):
       target_feature=self.fea_extrator_f(target)
       #       target_feature=torch.nn.functional.softmax(target_feature, dim=1)
       target_feature=target_feature.cpu().detach().numpy()
       target_labels=np.argmax(target_labels.cpu().detach().numpy(),axis=1)
       colors1 = '#00CED1' #点的颜色
       colors2 = '#DC143C'
       colors3 = '#008000'
       colors4 = '#000000'
       area = np.pi * 4**2  # 点面积 
       if tsne==0:       
           print('error')
           return
       else:
           target_feature = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000).fit_transform(target_feature.astype('float32'))
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]] 
           x3=target_feature[np.where(target_labels==3)[0]] 
           plt.scatter(x0[:,0],x0[:,1], s=area, c=colors1, alpha=0.4)
           plt.scatter(x1[:,0],x1[:,1], s=area, c=colors2, alpha=0.4)
           plt.scatter(x2[:,0],x2[:,1], s=area, c=colors3, alpha=0.4)
           plt.scatter(x3[:,0],x3[:,1], s=area, c=colors4, alpha=0.4)
           plt.show()

   def get_parameters(self) -> List[Dict]:
       params = [
            {"params": self.fea_extrator_f.fc1.parameters(), "lr_mult": 1},
            {"params": self.fea_extrator_f.fc2.parameters(), "lr_mult": 1},
            {"params": self.classifier.parameters(), "lr_mult": 1},
                ]
       return params
   

class feature_extractor_att(nn.Module):
    def __init__(self,hidden_1,hidden_2,hidden_dim):
         super(feature_extractor_att,self).__init__()
         
         self.fc1=nn.Linear(hidden_dim*62,hidden_1)
         self.fc2=nn.Linear(hidden_1,hidden_2)
         self.dropout1 = nn.Dropout(p=0.25)
         self.dropout2 = nn.Dropout(p=0.25)

         self.hidden_size = hidden_dim
         self.W_w = nn.Parameter(torch.Tensor(hidden_dim,hidden_dim))
         self.u_w = nn.Parameter(torch.Tensor(hidden_dim,1))

         nn.init.uniform_(self.W_w,-0.1,0.1)
         nn.init.uniform_(self.u_w,-0.1,0.1)

    def forward(self,x):
         # batch_size*seq_len*emd_dim
         
         # attention
         score = torch.tanh(torch.matmul(x, self.W_w))# batch_size*seq_len*emd_dim
         attention_weights = F.softmax(torch.matmul(score, self.u_w),dim=1)

         score_x = x * attention_weights

         #
         score_x = score_x.reshape(-1, 62*self.hidden_size)
         score_x = self.fc1(score_x)
         score_x = F.relu(score_x)
         score_x = self.fc2(score_x)

         score_x = F.relu(score_x)

         return score_x
    

class Dann_withatt(nn.Module):
   def __init__(self,hidden_1,hidden_2,num_of_class,max_iter,hidden_dim):
       super(Dann_withatt,self).__init__()
       self.fea_extrator_f= feature_extractor_att(hidden_1,hidden_2,hidden_dim)
       self.max_iter=max_iter
    #    self.upper_threshold=upper_threshold
    #    self.lower_threshold=lower_threshold
       self.classifier=nn.Linear(hidden_2,num_of_class)
    #    self.classifier1=classifier_transformer(hidden_2,num_of_class)
#       self.diff=(upper_threshold-lower_threshold)
    #    self.threshold=upper_threshold
    #    self.cluster_label=np.zeros(num_of_class)
    #    self.num_of_class=num_of_class
   def forward(self,source,target,source_label):
       feature_source_f=self.fea_extrator_f(source)
       feature_target_f=self.fea_extrator_f(target)
       ##torch.matmul(source_label.T,torch.ones(batch_num,num_of_class))
       ## Update P through some algebra computations for the convenice of broadcast
       source_predict=self.classifier(feature_source_f)
       target_predict=self.classifier(feature_target_f)
#       source_logit  =source_predict
       source_label_feature=torch.nn.functional.softmax(source_predict, dim=1)
       target_label_feature=torch.nn.functional.softmax(target_predict, dim=1)
    #    ## DAC part
    #    sim_matrix=self.get_cos_similarity_distance(source_label_feature)
    #    sim_matrix_target=self.get_cos_similarity_distance(target_label_feature)
       return source_predict,feature_source_f,feature_target_f,source_label_feature,target_label_feature
#    def compute_target_centroid(self,target,target_label):
#        feature_source_g=self.fea_extrator_f(target)
#        target_centroid=torch.matmul(torch.inverse(torch.diag(target_label.sum(axis=0))+torch.eye(self.num_of_class).cuda()),torch.matmul(target_label.T,feature_source_g))
#        return target_centroid
   def target_domain_evaluation(self,test_features,test_labels):
       self.eval()
       feature_target_f=self.fea_extrator_f(test_features)
       test_logit=self.classifier(feature_target_f)
       test_cluster=torch.nn.functional.softmax(test_logit, dim=1)
       test_cluster=np.argmax(test_cluster.cpu().detach().numpy(),axis=1)
       test_labels=np.argmax(test_labels.cpu().detach().numpy(),axis=1)

       test_predict = test_cluster

    #    test_predict=np.zeros_like(test_labels)
    #    for i in range(len(self.cluster_label)):
    #        cluster_index=np.where(test_cluster==i)[0]
    #        test_predict[cluster_index]=self.cluster_label[i]
#       acc=np.sum(label_smooth(test_predict)==test_labels)/len(test_predict)
       acc=np.sum(test_predict==test_labels)/len(test_predict)
       nmi=metrics.normalized_mutual_info_score(test_predict,test_labels)
       return acc,nmi   
   def cluster_label_update(self,source_features,source_labels):
       self.eval()
       feature_target_f=self.fea_extrator_f(source_features)
       source_logit=self.classifier(feature_target_f)
       source_cluster=np.argmax(torch.nn.functional.softmax(source_logit, dim=1).cpu().detach().numpy(),axis=1)
       source_labels=np.argmax(source_labels.cpu().detach().numpy(),axis=1)
    


       source_predict = source_cluster

       acc=np.sum(source_predict==source_labels)/len(source_predict)
       nmi=metrics.normalized_mutual_info_score(source_predict,source_labels)
       return acc,nmi
   def visualization(self,target,target_labels,tsne=0):
       feature_target_f=self.fea_extrator_f(target)
       target_feature=self.classifier(feature_target_f)
       #       target_feature=torch.nn.functional.softmax(target_feature, dim=1)
       target_feature=target_feature.cpu().detach().numpy()
       feature_target_f=feature_target_f.cpu().detach().numpy()
       target_labels=np.argmax(target_labels.cpu().detach().numpy(),axis=1)
       colors1 = '#00CED1' #点的颜色
       colors2 = '#DC143C'
       colors3 = '#008000'
       area = np.pi * 4**2  # 点面积 
       if tsne==0:       
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]]
       # 画散点图
           fig = plt.figure()
           ax = Axes3D(fig)
           ax.scatter(x0[:,0],x0[:,1],x0[:,2], s=area, c=colors1, alpha=0.4)
           ax.scatter(x1[:,0],x1[:,1],x1[:,2], s=area, c=colors2, alpha=0.4)
           ax.scatter(x2[:,0],x2[:,1],x2[:,2], s=area, c=colors3, alpha=0.4)
           plt.show()
       else:
           target_feature = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000).fit_transform(feature_target_f.astype('float32'))
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]] 
           plt.scatter(x0[:,0],x0[:,1], s=area, c=colors1, alpha=0.4)
           plt.scatter(x1[:,0],x1[:,1], s=area, c=colors2, alpha=0.4)
           plt.scatter(x2[:,0],x2[:,1], s=area, c=colors3, alpha=0.4)
           plt.show()
   def visualization_4(self,target,target_labels,tsne=0):
       target_feature=self.fea_extrator_f(target)
       #       target_feature=torch.nn.functional.softmax(target_feature, dim=1)
       target_feature=target_feature.cpu().detach().numpy()
       target_labels=np.argmax(target_labels.cpu().detach().numpy(),axis=1)
       colors1 = '#00CED1' #点的颜色
       colors2 = '#DC143C'
       colors3 = '#008000'
       colors4 = '#000000'
       area = np.pi * 4**2  # 点面积 
       if tsne==0:       
           print('error')
           return
       else:
           target_feature = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000).fit_transform(target_feature.astype('float32'))
           x0=target_feature[np.where(target_labels==0)[0]]
           x1=target_feature[np.where(target_labels==1)[0]]
           x2=target_feature[np.where(target_labels==2)[0]] 
           x3=target_feature[np.where(target_labels==3)[0]] 
           plt.scatter(x0[:,0],x0[:,1], s=area, c=colors1, alpha=0.4)
           plt.scatter(x1[:,0],x1[:,1], s=area, c=colors2, alpha=0.4)
           plt.scatter(x2[:,0],x2[:,1], s=area, c=colors3, alpha=0.4)
           plt.scatter(x3[:,0],x3[:,1], s=area, c=colors4, alpha=0.4)
           plt.show()

   def get_parameters(self) -> List[Dict]:
       params = [
            {"params": self.fea_extrator_f.fc1.parameters(), "lr_mult": 1},
            {"params": self.fea_extrator_f.fc2.parameters(), "lr_mult": 1},
            {"params": self.classifier.parameters(), "lr_mult": 1},
                ]
       return params


# class Dann_withLSTM_attention(nn.Module):
#     def __init__(self,hidden_1,hidden_2,num_of_class,max_iter,lstm_input_dim=5, lstm_hidden_dim=32, lstm_num_layers=2):
#         super(Dann_withLSTM_attention,self).__init__()
#         self.fea_extrator_f= feature_extractor_LSTM_noLinear(hidden_1,hidden_2,lstm_input_dim,lstm_hidden_dim,lstm_num_layers)
#         self.max_iter=max_iter
#         #    self.upper_threshold=upper_threshold
#         #    self.lower_threshold=lower_threshold
#         self.classifier=nn.Linear(62*32,num_of_class)
#         # self.classifier=classifier_transformer(hidden_2,num_of_class)
#         # self.att = nn.MultiheadAttention(embed_dim=lstm_hidden_dim,num_heads=1,batch_first=True,dropout=0.5)
#         self.att = Attention(32)
#     #       self.diff=(upper_threshold-lower_threshold)
#         #    self.threshold=upper_threshold
#         #    self.cluster_label=np.zeros(num_of_class)
#         #    self.num_of_class=num_of_class

#         self.lstm_hidden_dim = lstm_hidden_dim

#     def forward(self,source,target,source_label):
#         feature_source_f=self.fea_extrator_f(source)
#         feature_target_f=self.fea_extrator_f(target)

#         # 把用于域适应的压扁
#         feature_source_f_ = feature_source_f.reshape(-1, 62*self.lstm_hidden_dim)
#         feature_target_f_ = feature_target_f.reshape(-1, 62*self.lstm_hidden_dim)

#         ##torch.matmul(source_label.T,torch.ones(batch_num,num_of_class))
#         ## Update P through some algebra computations for the convenice of broadcast

#         # A = self.att(feature_source_f)
#         # A_target = self.att(feature_target_f)

#         # source_predict=self.classifier(feature_source_f)
#         # target_predict=self.classifier(feature_target_f)

#         # batch_size = A.shape[0]

#         # A = A.reshape(batch_size, -1)
#         # A_target = A_target.reshape(batch_size, -1)

#         source_predict=self.classifier(feature_source_f_)
#         target_predict=self.classifier(feature_target_f_)

#     #       source_logit  =source_predict
#         source_label_feature=torch.nn.functional.softmax(source_predict, dim=1)
#         target_label_feature=torch.nn.functional.softmax(target_predict, dim=1)
#         #    ## DAC part
#         #    sim_matrix=self.get_cos_similarity_distance(source_label_feature)
#         #    sim_matrix_target=self.get_cos_similarity_distance(target_label_feature)
#         return source_predict,feature_source_f_,feature_target_f_,source_label_feature,target_label_feature
#     #    def compute_target_centroid(self,target,target_label):
#     #        feature_source_g=self.fea_extrator_f(target)
#     #        target_centroid=torch.matmul(torch.inverse(torch.diag(target_label.sum(axis=0))+torch.eye(self.num_of_class).cuda()),torch.matmul(target_label.T,feature_source_g))
#     #        return target_centroid
#     def target_domain_evaluation(self,test_features,test_labels):
#         self.eval()
#         feature_target_f=self.fea_extrator_f(test_features)
#         test_logit=self.classifier(feature_target_f)
#         test_cluster=torch.nn.functional.softmax(test_logit, dim=1)
#         test_cluster=np.argmax(test_cluster.cpu().detach().numpy(),axis=1)
#         test_labels=np.argmax(test_labels.cpu().detach().numpy(),axis=1)

#         test_predict = test_cluster

#         #    test_predict=np.zeros_like(test_labels)
#         #    for i in range(len(self.cluster_label)):
#         #        cluster_index=np.where(test_cluster==i)[0]
#         #        test_predict[cluster_index]=self.cluster_label[i]
#     #       acc=np.sum(label_smooth(test_predict)==test_labels)/len(test_predict)
#         acc=np.sum(test_predict==test_labels)/len(test_predict)
#         nmi=metrics.normalized_mutual_info_score(test_predict,test_labels)
#         return acc,nmi   
#     def cluster_label_update(self,source_features,source_labels):
#         self.eval()
#         feature_target_f=self.fea_extrator_f(source_features)
#         source_logit=self.classifier(feature_target_f)
#         source_cluster=np.argmax(torch.nn.functional.softmax(source_logit, dim=1).cpu().detach().numpy(),axis=1)
#         source_labels=np.argmax(source_labels.cpu().detach().numpy(),axis=1)
#         #    for i in range(len(self.cluster_label)):
#         #        samples_in_cluster_index=np.where(source_cluster==i)[0]
#         #        label_for_samples=source_labels[samples_in_cluster_index]
#         #        if len(label_for_samples)==0:
#         #           self.cluster_label[i]=0
#         #        else:
#         #           label_for_current_cluster=np.argmax(np.bincount(label_for_samples))
#         #           self.cluster_label[i]=label_for_current_cluster
#         #    source_predict=np.zeros_like(source_labels)
#         #    for i in range(len(self.cluster_label)):
#         #        cluster_index=np.where(source_cluster==i)[0]
#         #        source_predict[cluster_index]=self.cluster_label[i]


#         source_predict = source_cluster

#         acc=np.sum(source_predict==source_labels)/len(source_predict)
#         nmi=metrics.normalized_mutual_info_score(source_predict,source_labels)
#         return acc,nmi
#     def visualization(self,target,target_labels,tsne=0):
#         feature_target_f=self.fea_extrator_f(target)
#         target_feature=self.classifier(feature_target_f)
#         #       target_feature=torch.nn.functional.softmax(target_feature, dim=1)
#         target_feature=target_feature.cpu().detach().numpy()
#         feature_target_f=feature_target_f.cpu().detach().numpy()
#         target_labels=np.argmax(target_labels.cpu().detach().numpy(),axis=1)
#         colors1 = '#00CED1' #点的颜色
#         colors2 = '#DC143C'
#         colors3 = '#008000'
#         area = np.pi * 4**2  # 点面积 
#         if tsne==0:       
#             x0=target_feature[np.where(target_labels==0)[0]]
#             x1=target_feature[np.where(target_labels==1)[0]]
#             x2=target_feature[np.where(target_labels==2)[0]]
#         # 画散点图
#             fig = plt.figure()
#             ax = Axes3D(fig)
#             ax.scatter(x0[:,0],x0[:,1],x0[:,2], s=area, c=colors1, alpha=0.4)
#             ax.scatter(x1[:,0],x1[:,1],x1[:,2], s=area, c=colors2, alpha=0.4)
#             ax.scatter(x2[:,0],x2[:,1],x2[:,2], s=area, c=colors3, alpha=0.4)
#             plt.show()
#         else:
#             target_feature = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000).fit_transform(feature_target_f.astype('float32'))
#             x0=target_feature[np.where(target_labels==0)[0]]
#             x1=target_feature[np.where(target_labels==1)[0]]
#             x2=target_feature[np.where(target_labels==2)[0]] 
#             plt.scatter(x0[:,0],x0[:,1], s=area, c=colors1, alpha=0.4)
#             plt.scatter(x1[:,0],x1[:,1], s=area, c=colors2, alpha=0.4)
#             plt.scatter(x2[:,0],x2[:,1], s=area, c=colors3, alpha=0.4)
#             plt.show()
#     def visualization_4(self,target,target_labels,tsne=0):
#         target_feature=self.fea_extrator_f(target)
#         #       target_feature=torch.nn.functional.softmax(target_feature, dim=1)
#         target_feature=target_feature.cpu().detach().numpy()
#         target_labels=np.argmax(target_labels.cpu().detach().numpy(),axis=1)
#         colors1 = '#00CED1' #点的颜色
#         colors2 = '#DC143C'
#         colors3 = '#008000'
#         colors4 = '#000000'
#         area = np.pi * 4**2  # 点面积 
#         if tsne==0:       
#             print('error')
#             return
#         else:
#             target_feature = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000).fit_transform(target_feature.astype('float32'))
#             x0=target_feature[np.where(target_labels==0)[0]]
#             x1=target_feature[np.where(target_labels==1)[0]]
#             x2=target_feature[np.where(target_labels==2)[0]] 
#             x3=target_feature[np.where(target_labels==3)[0]] 
#             plt.scatter(x0[:,0],x0[:,1], s=area, c=colors1, alpha=0.4)
#             plt.scatter(x1[:,0],x1[:,1], s=area, c=colors2, alpha=0.4)
#             plt.scatter(x2[:,0],x2[:,1], s=area, c=colors3, alpha=0.4)
#             plt.scatter(x3[:,0],x3[:,1], s=area, c=colors4, alpha=0.4)
#             plt.show()
#     #    def get_cos_similarity_distance(self, features):
#     #         """Get distance in cosine similarity
#     #         :param features: features of samples, (batch_size, num_clusters)
#     #         :return: distance matrix between features, (batch_size, batch_size)
#     #         """
#     #         # (batch_size, num_clusters)
#     #         features_norm = torch.norm(features, dim=1, keepdim=True)
#     #         # (batch_size, num_clusters)
#     #         features = features / features_norm
#     #         # (batch_size, batch_size)
#     #         cos_dist_matrix = torch.mm(features, features.transpose(0, 1))
#     #         return cos_dist_matrix
#     #    def get_cos_similarity_by_threshold(self, cos_dist_matrix):
#     #         """Get similarity by threshold
#     #         :param cos_dist_matrix: cosine distance in matrix,
#     #         (batch_size, batch_size)
#     #         :param threshold: threshold, scalar
#     #         :return: distance matrix between features, (batch_size, batch_size)
#     #         """
#     #         # 测试集的相似度矩阵由源域的相似度矩阵取阈值二值化确定
#     #         device = cos_dist_matrix.device
#     #         dtype = cos_dist_matrix.dtype
#     #         similar = torch.tensor(1, dtype=dtype, device=device)
#     #         dissimilar = torch.tensor(0, dtype=dtype, device=device)
#     #         sim_matrix = torch.where(cos_dist_matrix > self.threshold, similar,
#     #                                  dissimilar)
#     #         return sim_matrix
#     #    def compute_indicator(self,cos_dist_matrix):
#     #        device = cos_dist_matrix.device
#     #        dtype = cos_dist_matrix.dtype
#     #        selected = torch.tensor(1, dtype=dtype, device=device)
#     #        not_selected = torch.tensor(0, dtype=dtype, device=device)
#     #        w2=torch.where(cos_dist_matrix < self.lower_threshold,selected,not_selected)
#     #        w1=torch.where(cos_dist_matrix > self.upper_threshold,selected,not_selected)
#     #        w = w1 + w2
#     #        nb_selected=torch.sum(w)
#     #        return w,nb_selected
#     #    def update_threshold(self, epoch: int):
#     #         """Update threshold
#     #         :param threshold: scalar
#     #         :param epoch: scalar
#     #         :return: new_threshold: scalar
#     #         """
#     #         n_epochs = self.max_iter
#     #         diff = self.upper_threshold - self.lower_threshold
#     #         eta = diff / n_epochs
#     # #        eta=self.diff/ n_epochs /2
#     #         # First epoch doesn't update threshold
#     #         if epoch != 0:
#     #             self.upper_threshold = self.upper_threshold-eta
#     #             self.lower_threshold = self.lower_threshold+eta
#     #         else:
#     #             self.upper_threshold = self.upper_threshold
#     #             self.lower_threshold = self.lower_threshold
#     #         self.threshold=(self.upper_threshold+self.lower_threshold)/2
#     # #        print(">>> new threshold is {}".format(new_threshold), flush=True)
#     def get_parameters(self) -> List[Dict]:
#         params = [
#                 {"params": self.fea_extrator_f.fc1.parameters(), "lr_mult": 1},
#                 {"params": self.fea_extrator_f.fc2.parameters(), "lr_mult": 1},
#                 {"params": self.classifier.parameters(), "lr_mult": 1},
#                     ]
#         return params
