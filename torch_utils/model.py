# Import necessary libraries
import torch
import pytorch_lightning as pl
#NCODLoss has manual optmization as written here https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html# according
#to the paper https://github.com/RSTLess-research/NCOD-Learning-with-noisy-labels/tree/main

# Define the BaseNN class
class BaseNN(pl.LightningModule):
    #TODO: DEFINITION OF INPUTS
    def __init__(self, main_module, loss, optimizer, metrics={}, log_params={},
                 step_routing = {"model_input_from_batch":[0],
                                 "loss_input_from_batch": [1], "loss_input_from_model_output": [0],
                                 "metrics_input_from_batch": [1], "metrics_input_from_model_output": [0]},
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

        # Define how batch and model output are routed to model, loss and metrics
        self.step_routing = step_routing

        # Define a custom logging function
        self.custom_log = lambda name, value: self.log(name, value, **log_params)

    # Define the forward pass of the neural network
    def forward(self, *args, **kwargs):
        return self.main_module(*args, **kwargs)

    # Configure the optimizer for training
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())   
        return optimizer

    def on_epoch_end(self):
        # Step through each scheduler
        for scheduler in self.lr_schedulers():
            scheduler.step()

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
            metric_values = self.compute_metrics(batch, self.step_routing["metrics_input_from_batch"],
                                                model_output, self.step_routing["metrics_input_from_model_output"],
                                                split_name)

        #TODO: return loss is correct?
        print("LOSS",loss.item())
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
            if isinstance(keys, int) or isinstance(keys, str):
                keys = [keys]
            if isinstance(keys, list):
                input_args += [object[i] for i in keys]
            elif isinstance(keys, dict):
                for k,i in keys.items():
                    if i is None:
                        input_kwargs[k] = object
                    else:
                        input_kwargs[k] = object[i]
            elif keys is None:
                input_args.append(object)
            else:
                raise NotImplementedError("keys type not recognized")
        return input_args, input_kwargs

    def compute_loss(self, batch, loss_input_from_batch, model_output, loss_input_from_model_output, split_name):
        if isinstance(self.loss, dict):
            loss = 0
            for loss_name, loss_func in self.loss.items():
                # TODO: WEIGHT LOSS
                loss += self._compute(loss_name, loss_func, batch, loss_input_from_batch, model_output, loss_input_from_model_output, split_name)
            self.custom_log(split_name+'_loss', loss)
        else:
            loss = self._compute("loss", self.loss, batch, loss_input_from_batch, model_output, loss_input_from_model_output, split_name)

        return loss
    
    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             param_norm = param.grad.data.max()  # Calculate L2 norm of gradients
    #             print(name,param_norm)
    #     # norms = pl.utilities.grad_norm(self.layer, norm_type=2)
    #     # self.log_dict(norms)
    #     # print(norms)
    
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
    
    def compute_metrics(self, batch, metrics_input_from_batch, model_output, metrics_input_from_model_output, split_name):
        metric_values = {}
        for metric_name, metric_func in self.metrics.items():
            metric_values[metric_name] = self._compute(metric_name, metric_func, batch, metrics_input_from_batch, model_output, metrics_input_from_model_output, split_name)
        return metric_values
    
    def _compute(self, name, func, batch, input_from_batch, model_output, input_from_model_output, split_name):
        # If metrics_input is a dictionary, routing is different for each metric
        if isinstance(input_from_batch, dict) and name in input_from_batch:
            app1 = input_from_batch[name]
        else:
            app1 = input_from_batch
        if isinstance(input_from_model_output, dict) and name in input_from_model_output:
            app2 = input_from_model_output[name]
        else:
            app2 = input_from_model_output

        input_args, input_kwargs = self.get_input_args_kwargs((batch, app1), (model_output, app2))

        value = func(*input_args,**input_kwargs)

        if isinstance(value, dict):
            for key in value:
                self.custom_log(split_name+'_'+name+'_'+key, value[key])
        else:
            self.custom_log(split_name+'_'+name, value)

        return value

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

def get_torchvision_model_as_decoder(example_datum, *args, **kwargs):
    forward_model = torchvision_utils.get_torchvision_model(*args, **kwargs)
    inverted_model = torchvision_utils.invert_model(forward_model, example_datum)
    return inverted_model

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
