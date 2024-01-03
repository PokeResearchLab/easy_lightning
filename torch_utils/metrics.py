import torch.nn.functional as F
import torchmetrics
import torch

# Function to compute accuracy for neural network predictions
def nn_accuracy(y_hat, y):
    # Apply softmax to predictions and get the class with the highest probability
    soft_y_hat = F.softmax(y_hat).argmax(dim=-1)
    soft_y = y.argmax(dim=-1)
    
    # Calculate accuracy by comparing predicted and actual class labels
    acc = (soft_y_hat.int() == soft_y.int()).float().mean()
    return acc

# Custom Accuracy to compute accuracy with Soft Labels as a torch.Module
class SoftLabelsAccuracy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        # Calculate accuracy by comparing predicted and actual class labels
        return (preds.argmax(dim=1) == target.argmax(dim=1)).float().mean()

# Custom Accuracy to compute accuracy with Soft Labels as a torchmetrics.Metric
class SoftLabelsAccuracy2(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        # Initialize state variables for correct predictions and total examples
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Update correct predictions and total examples
        self.correct += torch.sum(preds.argmax(dim=1) == target.argmax(dim=1))
        self.total += target.shape[0]

    def compute(self):
        # Compute accuracy as the ratio of correct predictions to total examples
        return self.correct.float() / self.total

# class NDCG(torchmetrics.Metric):
#     def __init__(self, top_k=10):
#         super().__init__()
#         self.top_k = top_k
        
#         # Initialize state variables for correct predictions and total examples
#         self.add_state("correct", default=torch.tensor(0.), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")

#     def update(self, scores: torch.Tensor, relevance: torch.Tensor):
#         # Update values
#         ordered_items = scores.argsort(dim=-1, descending=True)
#         ranks = ordered_items.argsort(dim=-1)+1

#         app = torch.log(ranks+1)
#         dcg = ((ranks<=self.top_k)*relevance/app).sum(-1)
#         idcg = (1/torch.log(torch.arange(1,min(self.top_k,scores.shape[1])+1)+1)).sum(-1)
#         ndcg = dcg/idcg

#         self.correct += ndcg.mean(-1).sum()
#         self.total += scores.shape[0]

#     def compute(self):
#         # Compute accuracy as the ratio of correct predictions to total examples
#         print("NDCG:",self.correct.float() / self.total)
#         return self.correct.float() / self.total
    
# class NDCG(torchmetrics.Metric):
#     def __init__(self, top_k=[5,10,20]):
#         super().__init__()
#         self.top_k = top_k if isinstance(top_k, list) else [top_k]
        
#         # Initialize state variables for correct predictions and total examples
#         for top_k in self.top_k:
#             self.add_state(f"correct@{top_k}", default=torch.tensor(0.), dist_reduce_fx="sum")
#         self.add_state(f"total", default=torch.tensor(0.), dist_reduce_fx="sum")

#     def update(self, scores: torch.Tensor, relevance: torch.Tensor):
#         # Update values
#         ordered_items = scores.argsort(dim=-1, descending=True)
#         ranks = ordered_items.argsort(dim=-1)+1
        
#         app = torch.log(ranks+1)
#         for top_k in self.top_k:
#             dcg = ((ranks<=top_k)*relevance/app).sum(-1)
#             idcg = (1/torch.log(torch.arange(1,min(top_k,scores.shape[-1])+1)+1)).sum(-1)
#             ndcg = dcg/idcg # ndcg.shape = (num_samples, lookback)
#             setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + ndcg.mean(-1).sum())
#         self.total += relevance.shape[0]

#     def compute(self):
#         # Compute accuracy as the ratio of correct predictions to total examples
#         out = {}
#         for k in self.top_k:
#             out[f"@{k}"] = getattr(self, f"correct@{k}") / self.total
#         #print(out)
#         return out
    
class RecMetric(torchmetrics.Metric):
    def __init__(self, top_k=[5,10,20]):
        super().__init__()
        self.top_k = top_k if isinstance(top_k, list) else [top_k]

        # Initialize state variables for correct predictions and total examples
        for top_k in self.top_k:
            self.add_state(f"correct@{top_k}", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state(f"total", default=torch.tensor(0.), dist_reduce_fx="sum")

    def compute(self):
        # Compute accuracy as the ratio of correct predictions to total examples
        out = {}
        for k in self.top_k:
            out[f"@{k}"] = getattr(self, f"correct@{k}") / self.total
        return out
    
class NDCG(RecMetric):
    def __init__(self, top_k=[5,10,20]):
        super().__init__(top_k)

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1

        app = torch.log(ranks+1)
        for top_k in self.top_k:
            dcg = ((ranks<=top_k)*relevance/app).sum(-1)
            k = min(top_k,scores.shape[-1])
            sorted_k_relevance = relevance.sort(dim=-1, descending=True).values[...,:k] #get first k items in sorted_relevance on last dimension  
            idcg = (sorted_k_relevance/torch.log(torch.arange(1,k+1)+1)).sum(-1)
            ndcg = dcg/idcg # ndcg.shape = (num_samples, lookback)
            setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + ndcg.mean(-1).sum())
        self.total += relevance.shape[0]
    
class MRR(RecMetric):
    def __init__(self, top_k=[5,10,20]):
        super().__init__(top_k)

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1

        relevant = relevance>0
        for top_k in self.top_k:
            # if len(ranks) == 0:
            #     mrr = 0
            # else:
            mrr = ((ranks<=top_k)*relevant*(1/ranks)).max(-1).values
            setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + mrr.sum())
        self.total += relevance.shape[0]

#TODO: implementare altre metriche

