# Import necessary libraries
import torch
import pytorch_lightning as pl
from .losses import NCODLoss
import torch.optim as optim
#NCODLoss has manual optmization as written here https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html# according
#to the paper https://github.com/RSTLess-research/NCOD-Learning-with-noisy-labels/tree/main

# Define the BaseNN class
class BaseNN(pl.LightningModule):
    #TODO: DEFINITION OF INPUTS
    def __init__(self, main_module, loss, optimizer, metrics={}, log_params={},
                 step_routing = {"model_input_from_batch":[0],
                                 "loss_input_from_batch": [1], "loss_input_from_model_output": [0],
                                 "metric_input_from_batch": [1], "metric_input_from_model_output": [0]},
                 **kwargs):
        super().__init__()

        # Store the main neural network module
        self.main_module = main_module

        # Store the optimizer function
        self.optimizer = optimizer

        # Store the primary loss function
        self.loss = loss
        
        # Define the metrics to be used for evaluation
        self.metrics = metrics

        self.step_routing = step_routing
        
        #TODO: Check this part if still needed
        # Prototype for customizing logging for multiple losses (if needed)
        # self.losses = loss
        # self.loss_log_params = {}
        # self.loss_weights = {}
        # for loss_name,loss_obj in self.losses.items():
        #     if isinstance(loss_obj, dict):
        #         self.losses[loss_name] = loss_obj["loss"]
        #         self.loss_log_params[loss_name] = loss_obj.get("log_params", {})
        #         self.loss_weights[loss_name] = loss_obj.get("weight", 1.0)
        #     else:
        #         self.losses[loss_name] = loss_obj
        #         self.loss_log_params[loss_name] = {}
        #         self.loss_weights[loss_name] = 1.0

        # Prototype for customizing logging for multiple metrics (if needed)
        # self.metrics = {}
        # self.metrics_log_params = {}
        # for metric_name, metric_obj in self.metrics.items():
        #     if isinstance(metric_obj, dict):
        #         self.metrics[metric_name] = metric_obj["metric"]
        #         self.metrics_log_params[metric_name] = metric_obj.get("log_params", {})
        #     else:
        #         self.metrics[metric_name] = metric_obj
        #         self.metrics_log_params[metric_name] = {}

        # Define a custom logging function
        self.custom_log = lambda name, value: self.log(name, value, **log_params)

    # Define the forward pass of the neural network
    def forward(self, *args, **kwargs):
        return self.main_module(*args, **kwargs)

    # Configure the optimizer for training
    def configure_optimizers(self):
        if isinstance(self.loss, NCODLoss):
            optimizer1 = self.optimizer(self.main_module.parameters())
            optimizer2 = optim.SGD(self.loss.parameters(), lr=0.1)
            # Define learning rate schedulers
            scheduler1 = {
                'scheduler': optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[80, 120], gamma=0.1),
                'interval': 'epoch',
                'frequency': 1
            }
            print("USING OPTIMIZERS FOR NCOD_LOSSS...")
            return [optimizer1, optimizer2], [scheduler1]
            
        optimizer1 = self.optimizer(self.parameters())   
        return optimizer1

    def on_epoch_end(self):
        # Step through each scheduler
        for scheduler in self.lr_schedulers():
            scheduler.step()

    #TODO: check if new step function is correct
    #TODO: nome adatto per model_input_def
    # Define a step function for processing a batch

    def step(self, batch, batch_idx, dataloader_idx, split_name):
        #TODO: what to do with batch_idx and dataloader_idx?
        model_output = self.compute_model_output(batch, self.step_routing["model_input_from_batch"])

        if self.loss is not None:
            loss = self.compute_loss(batch, self.step_routing["loss_input_from_batch"],
                                     model_output, self.step_routing["loss_input_from_model_output"],
                                     split_name)

        #TODO: should return metric_values?
        if len(self.metrics)>0:
            metric_values = self.compute_metrics(batch, self.step_routing["metric_input_from_batch"],
                                                model_output, self.step_routing["metric_input_from_model_output"],
                                                split_name)

        #TODO: return loss is correct?
        return loss

    def compute_model_output(self, batch, model_input_from_batch):
        model_input_args, model_input_kwargs = self.get_input_args_kwargs((batch, model_input_from_batch))

        model_output = self(*model_input_args, **model_input_kwargs)
        
        # if model_input_from_batch is None or (len(model_input_from_batch)==1 and not isinstance(batch,list)): #leave batch as is
        #     model_output = self(batch)
        # elif isinstance(model_input_from_batch, list):
        #     model_input = [batch[i] for i in model_input_from_batch]
        #     model_output = self(*model_input)
        # elif isinstance(model_input_from_batch, dict):
        #     model_input = {k:batch[v] for k,v in model_input_from_batch.items()}
        #     model_output = self(**model_input)
        # else:
        #     raise NotImplementedError("model_input_from_batch not recognized")
        return model_output
    
    def get_input_args_kwargs(self, *args):
        input_args, input_kwargs = [],{}
        for object,keys in args:
            if isinstance(keys, list):
                input_args += [object[i] for i in keys]
            elif isinstance(keys, dict):
                for k,i in keys.items():
                    input_kwargs[k] = object[i]
            elif keys is None:
                input_args.append(object)
            else:
                raise NotImplementedError("keys type not recognized")
        return input_args, input_kwargs

    def compute_loss(self, batch, loss_input_from_batch, model_output, loss_input_from_model_output, split_name):
        loss_input_args, loss_input_kwargs = self.get_input_args_kwargs((batch, loss_input_from_batch), (model_output, loss_input_from_model_output))

        loss = self.loss(*loss_input_args, **loss_input_kwargs)

        self.custom_log(split_name+'_loss', loss)

        return loss
    
    #sum of different losses? or left to user?
    # if sum done by us, we could log each loss separately
    # if sum done by user, we could only log the total loss
    # ....
    # TODO: più loss?
    # Come are se le loss hanno diversi inputs?
    # loss_input_from_model_output diventa un dizionario di losses
    # total_loss = 0
    # for loss_name, loss_obj in self.losses.items():
    #     loss = loss_obj(model_output, batch)
    #     total_loss += loss
    #     self.custom_log(split_name+'_'+loss_name, loss)
    # self.custom_log(split_name+'_total_loss', total_loss)
    
    def compute_metrics(self, batch, metric_input_from_batch, model_output, metric_input_from_model_output, split_name):
        for metric_name, metric_func in self.metrics.items():
            # If metric_input is a dictionary, routing is different for each metric
            if isinstance(metric_input_from_batch, dict) and metric_name in metric_input_from_batch:
                app1 = metric_input_from_batch[metric_name]
            else:
                app1 = metric_input_from_batch
            if isinstance(metric_input_from_model_output, dict) and metric_name in metric_input_from_model_output:
                app2 = metric_input_from_model_output[metric_name]
            else:
                app2 = metric_input_from_model_output

            metric_input_args, metric_input_kwargs = self.get_input_args_kwargs((batch, app1), (model_output, app2))

            metric_value = metric_func(*metric_input_args,**metric_input_kwargs)
            self.custom_log(split_name+'_'+metric_name, metric_value)    

    # Training step
    def training_step(self, batch, batch_idx, dataloader_idx=0): return self.step(batch, batch_idx, dataloader_idx, "train")

    # Validation step
    #why dataloader_idx=0?
    # TODO
    def validation_step(self, batch, batch_idx, dataloader_idx=0): return self.step(batch, batch_idx, dataloader_idx, "val")

    # Test step
    def test_step(self, batch, batch_idx, dataloader_idx=0): return self.step(batch, batch_idx, dataloader_idx, "test")
    
    # TODO: Predict step
    # def predict_step(self, batch, batch_idx, dataloader_idx): return self.step(batch, batch_idx, dataloader_idx, "predict")

# Define functions for getting and loading torchvision models
def get_torchvision_model(*args, **kwargs): return torchvision_utils.get_torchvision_model(*args, **kwargs)

def load_torchvision_model(*args, **kwargs): return torchvision_utils.load_torchvision_model(*args, **kwargs)

# Define an Identity module
class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# Define a LambdaLayer module
class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# Class MLP (Multi-Layer Perceptron) (commented out for now)
# class MLP(BaseNN):
#     def __init__(self, input_size, output_size, neurons_per_layer, activation_function=None, lr=None, loss = None, acc = None, **kwargs):
#         super().__init__()

#         layers = []
#         in_size = input_size
#         for out_size in neurons_per_layer:
#             layers.append(torch.nn.Linear(in_size, out_size))
#             if activation_function is not None:
#                 layers.append(getattr(torch.nn, activation_function)())
#             in_size = out_size
#         layers.append(torch.nn.Linear(in_size, output_size))
#         self.main_module = torch.nn.Sequential(*layers)

# Import additional libraries
from . import torchvision_utils  # put here otherwise circular import
