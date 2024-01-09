import torch.nn.functional as F
import torchmetrics
import torch
import numpy as np

#TODO: add metrics definition here
#class NDCG(torch.nn.Module):
# prendere da RecBole

class metrics(torch.nn.Module):
    def __init__(self, y_hat):
        super().__init__()
        self.metrics = torchmetrics.MetricCollection({
            "accuracy": torchmetrics.Accuracy(),
            "precision": torchmetrics.Precision(),
            "recall": torchmetrics.Recall(),
            "f1": torchmetrics.F1(),
            "ndcg": torchmetrics.NDCG(),
            "mrr": torchmetrics.MRR(),
            "auc": torchmetrics.AUC(),
            "map": torchmetrics.AveragePrecision(),
            "rmse": torchmetrics.MeanSquaredError(),
            "mae": torchmetrics.MeanAbsoluteError(),
            "r2": torchmetrics.R2Score(),
            "mse": torchmetrics.MeanSquaredError(),
            "mape": torchmetrics.MeanAbsolutePercentageError(),
            "rmsle": torchmetrics.MeanSquaredLogError(),
            "msle": torchmetrics.MeanSquaredLogError(),
            "soft_labels_accuracy": SoftLabelsAccuracy(),
            #Recommendation metrics
            "ndcg_at_k": NDCG(),
        })

        self.y_hat = y_hat
        scores = 4 #da dove la prendiamo?
        preds = scores.argsort(descending=True)
        ranks = preds.argsort()

        indexes = torch.arange(1, len(scores) + 1).unsqueeze(0) #?? controllare se è giusto per usare torchmetrics

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        # Calculate accuracy by comparing predicted and actual class labels
        #return (preds.argmax(dim=1) == target.argmax(dim=1)).float().mean()
        #return torchmetrics.functional.accuracy(preds, target)
        return self.metrics(preds, target)


class NDCG(torch.nn.Module):
    def __init__(self, y_hat):
        super().__init__()
        #???

        self.y_hat = y_hat

        self.target = 'listaordinatapositiveitems' #da dove la prendiamo?
        relevance_list = torch.ones(len(self.y_hat)) if relevance_list is None else relevance_list
     
    # lista di cose rilevanti: [] #ordinata target_y
    # lista di relevance: [] #ordinata #default = 1
    # relevance_list = torch.ones(len(target_y)) if relevance_list is None else relevance_list

    # scores: - lista ordinata di scores

    # parametro format="scores","preds","ranks": 
    # a seconda del formato, si trasforma
    # - True: se gli item sono già stati ordinati per score
    #         ---> lista ordinata di item predetti
    #         preds = scores
    # - False: se gli item non sono stati ordinati per score
    # preds = scores.argsort(descending=True)
    # ranks = preds.argsort()

    
    def dcg_at_k(self, scores, k):
        # Discounted Cumulative Gain at position k
        return np.sum((2 ** scores - 1) / np.log2(np.arange(2, len(scores) + 2)))[:k]


    def ndcg_at_k(self, target, k_values):
        # Normalized Discounted Cumulative Gain at position k
        ideal_dcg = self.dcg_at_k(sorted(target, reverse=True), max(k_values))
    
        ndcg_values = {}
        for k in k_values:
            actual_dcg = self.dcg_at_k(self.y_hat, k)
            
            if ideal_dcg == 0:
                ndcg_values[k] = 0
            else:
                ndcg_values[k] = actual_dcg / ideal_dcg
        return ndcg_values


    def forward(self, target: torch.Tensor, ks = [1,3,5,10,20,50,100]):
        # Calculate accuracy by comparing predicted and actual class labels
        #return (preds.argmax(dim=1) == target.argmax(dim=1)).float().mean()
        #return torchmetrics.functional.accuracy(preds, target)
        #return torchmetrics.functional.ndcg(preds, target, ks)
        return self.ndcg_at_k(target, ks)



# # Function to compute accuracy for neural network predictions
# Custom Accuracy to compute accuracy with Soft Labels as a torch.Module
# 
class SoftLabelsAccuracy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        # Calculate accuracy by comparing predicted and actual class labels
        return (preds.argmax(dim=1) == target.argmax(dim=1)).float().mean()