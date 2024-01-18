import torch
import torchmetrics

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

