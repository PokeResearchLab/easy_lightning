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
    
    def not_nan_subset(self, **kwargs):
        if "relevance" in kwargs:
            # Subset other args, kwargs where relevance is not nan
            relevance = kwargs["relevance"]
            is_not_nan_per_sample = ~torch.isnan(relevance).any(-1)
            kwargs = {k: v[is_not_nan_per_sample] for k, v in kwargs.items()}
            # This keeps just the last dimension, the others are collapsed

        return kwargs
    
class CustomNDCG(RecMetric):
    def __init__(self, top_k=[5,10,20]):
        super().__init__(top_k)

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        # Call not_nan_subset to subset scores, relevance where relevance is not nan
        kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
        scores, relevance = kwargs["scores"], kwargs["relevance"]

        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1

        app = torch.log2(ranks+1)
        for top_k in self.top_k:
            dcg = ((ranks<=top_k)*relevance/app).sum(-1)
            k = min(top_k,scores.shape[-1])
            sorted_k_relevance = relevance.sort(dim=-1, descending=True).values[...,:k] #get first k items in sorted_relevance on last dimension  
            idcg = (sorted_k_relevance/torch.log2(torch.arange(1,k+1,device=sorted_k_relevance.device)+1)).sum(-1)
            ndcg = dcg/idcg # ndcg.shape = (num_samples, lookback)
            setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + ndcg.sum())
        self.total += relevance.shape[0]
    
class CustomMRR(RecMetric):
    def __init__(self, top_k=[5,10,20]):
        super().__init__(top_k)

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        # Call not_nan_subset to subset scores, relevance where relevance is not nan
        kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
        scores, relevance = kwargs["scores"], kwargs["relevance"]

        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1

        relevant = relevance>0
        for top_k in self.top_k:
            mrr = ((ranks<=top_k)*relevant*(1/ranks)).max(-1).values
            setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + mrr.sum())
        self.total += relevance.shape[0]

class CustomPrecision(RecMetric):
    def __init__(self, top_k=[5,10,20]):
        super().__init__(top_k)

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        # Call not_nan_subset to subset scores, relevance where relevance is not nan
        kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
        scores, relevance = kwargs["scores"], kwargs["relevance"]

        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1

        relevant = relevance>0
        for top_k in self.top_k:
            precision = (ranks<=top_k)*relevant/top_k
            setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + precision.sum())
        self.total += relevance.shape[0]

class CustomRecall(RecMetric):
    def __init__(self, top_k=[5,10,20]):
        super().__init__(top_k)

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        # Call not_nan_subset to subset scores, relevance where relevance is not nan
        kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
        scores, relevance = kwargs["scores"], kwargs["relevance"]

        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1

        relevant = relevance>0
        for top_k in self.top_k:
            recall = (ranks<=top_k)*relevant/relevant.sum(-1,keepdim=True)#torch.minimum(relevant.sum(-1,keepdim=True),top_k*torch.ones_like(relevant.sum(-1,keepdim=True)))
            setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + recall.sum())
        self.total += relevance.shape[0]

class CustomF1(RecMetric):
    def __init__(self, top_k=[5,10,20]):
        super().__init__(top_k)
        self.precision = CustomPrecision(top_k)
        self.recall = CustomRecall(top_k)

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        self.precision.update(scores, relevance)
        self.recall.update(scores, relevance)

    def compute(self):
        precision = self.precision.compute()
        recall = self.recall.compute()
        out = {}
        for k in self.top_k:
            out[f"@{k}"] = 2*(precision[f"@{k}"]*recall[f"@{k}"])/(precision[f"@{k}"]+recall[f"@{k}"])
        return out

class CustomPrecisionWithRelevance(RecMetric):
    def __init__(self, top_k=[5,10,20]):
        super().__init__(top_k)

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        # Call not_nan_subset to subset scores, relevance where relevance is not nan
        kwargs = self.not_nan_subset(scores=scores, relevance=relevance)
        scores, relevance = kwargs["scores"], kwargs["relevance"]

        # Update values
        ordered_items = scores.argsort(dim=-1, descending=True)
        ranks = ordered_items.argsort(dim=-1)+1

        for top_k in self.top_k:
            precision = (ranks<=top_k)*relevance/(top_k*relevance.sum(-1,keepdim=True))
            setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + precision.sum())
        self.total += relevance.shape[0]

class CustomMAP(RecMetric):
    def __init__(self, top_k=[5,10,20]):
        super().__init__(top_k)

        self.precision_at_k = CustomPrecisionWithRelevance(list(range(1,torch.max(torch.tensor(self.top_k))+1)))

    def update(self, scores: torch.Tensor, relevance: torch.Tensor):
        self.precision_at_k.update(scores, relevance)

    def compute(self):
        for top_k in self.top_k:
            for k in range(1,top_k+1):
                setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}") + getattr(self.precision_at_k, f"correct@{k}"))
            setattr(self, f"correct@{top_k}", getattr(self, f"correct@{top_k}")/k)
        setattr(self,"total", getattr(self.precision_at_k, f"total"))
        return super().compute()

