# Compression Feature Introduction

<!-- md-trans-meta sourceCommit=1636cfee96b0ba60c4df4b1c1c31a2234cdf1e09 translatedAt=2026-10-16T07:54:56.075Z pushedAt=2026-11-05T01:12:03.412Z nl=2 -->

Model compression is a collection of techniques that reduce the number of parameters, computational cost, memory usage, and bandwidth requirements to achieve model miniaturization and inference acceleration while keeping model performance essentially unchanged.
The core methods of model compression include quantization, pruning, knowledge distillation, and low-rank decomposition.

Quantization converts high-precision data such as FP32/FP16/BF16 into low-precision data such as INT8/FP8/INT4/FP4. By reducing the number of bits per parameter, it lowers storage usage and memory bandwidth, while leveraging hardware low-precision compute units to accelerate inference.
Depending on the optimization target, quantization can be divided into weight quantization, activation quantization, and KV-cache quantization. For the conceptual explanation of the compression features mentioned above, see [Compression Concepts](./en/compression_concepts.md).

The AMCT tool also provides quantization capabilities for large language models (LLMs). For details, see [AMCT-PyTorch LLM Quantization Tool](./en/AMCT_Pytorch_LLM.md).

## AMCT Quantization Process

The quantization process is shown in the following figure:

![](./en/figures/quant.png "Call flow")

The blue part is implemented by the user, and the gray part is implemented by calling the APIs provided by AMCT. The brief process is as follows:

1. The user first constructs the original PyTorch model (ensure that the original model can perform inference normally) and the quantization configuration, and then calls the [quantize](./en/api/quantize.md) API to modify the original model and generate a quantization calibration model. During the modification, different data categories and quantization algorithms are used.

2. The user calls the [convert](./en/api/convert.md) API to convert the quantization calibration operators into the corresponding quantization deployment operators on the NPU, and outputs the quantization deployment model (of type **torch.nn.Module**).

## Call Example

The following is a code example of the key steps. It is for reference only and cannot be directly copied, compiled, or run.

```python
# 1. Import the AMCT package.
import amct_pytorch as amct

# 2. Call AMCT to quantize the model.
# 2.1 Generate the quantization calibration model.
# Build the network graph structure to be quantized.
ori_model = build_model()
model = copy.deepcopy(ori_model)
# Quantization configuration.
cfg = {
        'batch_num': 1,
        'quant_cfg': {
            'weights': {
                'type': 'int8',
                'symmetric': True,
                'strategy': 'tensor',
            },
        },
        'algorithm': {'minmax'},
      }
# Call the quantization API to generate the quantization calibration model.
amct.quantize(model, cfg)

# 2.2 Generate the quantization deployment model.
# Call the API to convert the quantization calibration model into the quantization deployment model.
amct.convert(model)
```

<details>
<summary><strong>More methods</strong></summary>

## Graph-Based Quantization

AMCT provides a graph-based quantization method for torch models. This method obtains the graph structure of a torch model by converting the torch model into an ONNX model, performs quantization optimization based on the parsed graph structure, and finally generates an ONNX quantized model.

### Post-Training Quantization

Post-training quantization is divided into [Accuracy-Based Automatic Quantization](#accuracy-based-automatic-quantization) and [Manual Quantization](#manual-quantization) depending on whether the quantization configuration file is manually tuned after quantization. For the quantization algorithm used in post-training quantization, see [algorithm introduction](./en/algorithm_brief.md).

#### Accuracy-Based Automatic Quantization

Accuracy-based automatic quantization is a feature provided for users who have certain requirements on quantization accuracy. While ensuring the model accuracy required by the user, this method automatically searches for the quantization configuration of the model and executes the post-training quantization process, ultimately generating a quantized model that meets the accuracy requirements.

The basic principle of accuracy-based automatic quantization is the same as that of manual quantization, but users do not need to manually adjust the quantization configuration file, which greatly simplifies the optimization process and improves quantization efficiency. For the layers and constraints supported by quantization, see [Layers and Constraints Supported by Post-Training Quantization](./en/api/create_quant_config.md#layers-and-constraints-supported-by-post-training-quantization).

##### Quantization Flow

The API call flow is shown in the following figure. The blue part is implemented by the user, and the gray part is implemented by calling the APIs provided by AMCT:
![](./en/figures/autocalibration.png "API call flow")

The main process is as follows:

1. <a name="li1620271314520"></a>Call [create\_quant\_config](./en/api/create_quant_config.md) to generate the quantization configuration file, and then call [accuracy\_based\_auto\_calibration](./en/api/accuracy_based_auto_calibration.md) to perform accuracy-based automatic quantization.
2. <a name="li23141824486"></a>Call the evaluator instance passed in by the user in [accuracy\_based\_auto\_calibration](./en/api/accuracy_based_auto_calibration.md) to test the accuracy and obtain the accuracy of the original model.

   This process also calls the quantization strategy module in [accuracy\_based\_auto\_calibration](./en/api/accuracy_based_auto_calibration.md) to output the initialized **quant config** file, which records that all layers can be quantized.

3. Use the initial quantization configuration file passed in by the user (generated by calling [create\_quant\_config](./en/api/create_quant_config.md) in [1](#li1620271314520)) to perform post-training quantization on the model and obtain the accuracy of the quantized **fake quant** model.
4. Compare the accuracy of the original model with that of the quantized **fake quant** model. If the accuracy meets the requirement, output the quantized deployment model and the **fake quant** model. If not, perform the accuracy-based automatic quantization process:
   1. Perform inference on the original PyTorch network, dump the input activation data of each layer, and cache it;
   2. Use the quantization factors from post-training quantization to construct a single-operator network for the quantization layer, and use the cached activation data to calculate the cosine similarity between the output of the quantized **fake quant** single-operator network and the output of the original PyTorch single-operator network.
   3. <a name="li411322192712"></a>Pass the list of cosine similarities to the quantization strategy module in [accuracy\_based\_auto\_calibration](./en/api/accuracy_based_auto_calibration.md). Based on the initialized quantization configuration file generated in [2](#li23141824486), the strategy module outputs a new **quant config** file after rolling back certain layers.
   4. Perform post-training quantization again based on the **quant config** file to obtain the rolled-back **fake quant** model.
   5. Call the evaluator module in [accuracy\_based\_auto\_calibration](./en/api/accuracy_based_auto_calibration.md) to perform accuracy testing on the rolled-back **fake quant** model and check whether the accuracy meets the requirement:
       - If the accuracy meets the requirement, output the rolled-back **fake quant** model and the deployment model.
       - If the accuracy does not meet the requirement, roll back the layer with the worst cosine similarity ranking, perform [4.c](#li411322192712) again, and output a new quantization configuration.
       - If the accuracy still does not meet the requirement after all layers are rolled back, no quantized model is generated.

The accuracy-based automatic quantization process inside the [accuracy\_based\_auto\_calibration](./en/api/accuracy_based_auto_calibration.md) API is shown in the following figure.

![](./en/figures/auto_quant.png "Automatic quantization process")

##### Call Example

This example demonstrates the accuracy-based automatic quantization process using AMCT. This process requires the user to implement a callback function that performs model inference to obtain accuracy. Since AMCT needs to filter quantization layers based on the accuracy data returned by the callback function, the value returned by the callback function should reflect the model accuracy as much as possible.

> [!NOTE] Note
>
> - In the following example, steps marked with "to be supplemented by the user" need to be supplemented by the user based on their own model and dataset. The example provides only sample code.
> - For the parts of the following example that call AMCT, adjust the function input parameters according to the actual situation.
> - The following are code examples of key steps. They cannot be directly copied, compiled, and run, and are for reference only.

```python
# 1. Import the AMCT package.
import os
import amct_pytorch as amct
from amct_pytorch.common.auto_calibration import AutoCalibrationEvaluatorBase

# 2. (To be supplemented by the user) Use the original model to be quantized and the test set to implement the callback functions calibration(), evaluate(), and metric_eval().
# The input parameters of the above callback functions must be consistent with those of the base class AutoCalibrationEvaluatorBase. Specifically:
# calibration() performs the inference for calibration.
# evaluate() performs the model accuracy test process.
# metric_eval() evaluates the accuracy loss between the original model and the quantized fake quant model. It returns True when the accuracy loss is less than the expected value, and False otherwise.
class ModelEvaluator(AutoCalibrationEvaluatorBase):
     # The evaluator for model
    def __init__(self, *args, **kwargs):
        # Initialize member variables.
        # Set the expected accuracy loss. Replace this with a specific value.
        self.diff = expected_acc_loss
        pass

    def calibration(self, model_file, weights_file):
        # Perform calibration inference on the model. The number of inference batches must be consistent with batch_num in the quantization configuration.
        pass

    def evaluate(self, model_file, weights_file):
        # evaluate the input models, get the eval metric of model
        pass

    def metric_eval(self, original_metric, new_metric):
        # Evaluate whether the accuracy loss between the original model and the quantized model meets expectations. If so, return True and the accuracy loss data; otherwise, return False and the accuracy loss data.
        loss = original_metric - new_metric
        if loss < self.diff:
            return True, loss
        return False, loss

# 3. (To be supplemented by the user.) Instantiate the PyTorch model to obtain the model object.
model = MyNet()

# 4. Call AMCT to perform accuracy-based automatic quantization.
# 4.1 Generate the quantization configuration.
config_json_file = './config.json'
skip_layers = []
batch_num = 1
amct.create_quant_config(config_json_file, model, input_data,
                          skip_layers, batch_num)

scale_offset_record_file = os.path.join(TMP, 'scale_offset_record.txt')
result_path = os.path.join(RESULT, 'model')

# 4.2 Initialize the evaluator.
evaluator = ModelEvaluator()

# 4.3 Perform automatic search of the accuracy-based quantization configuration.
amct.accuracy_based_auto_calibration(
        model=model,
        model_evaluator=evaluator,
        config_file=config_json_file,
        record_file=record_file,
        save_dir=result_path,
        input_data=input_data,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        strategy='BinarySearch',
        sensitivity='CosineSimilarity'
    )
```

#### Manual Quantization

##### Quantization Flow

The following figure shows the API call flow of post-training quantization:

![](./en/figures/ptq_interface.png "Post-training quantization API call flow")

The blue parts are implemented by the user, and the gray parts are implemented by calling the APIs provided by AMCT. The tool is used in the following scenarios:

1. The user first constructs the original PyTorch model, and then uses [create\_quant\_config](./en/api/create_quant_config.md) to generate the quantization configuration file.
2. Based on the PyTorch model and the quantization configuration file, the user can call the [quantize\_model](./en/api/quantize_model.md) API to optimize the original PyTorch model. The optimized PyTorch model contains the quantization algorithm.
3. Use the calibration set to perform forward inference in the PyTorch environment to generate quantization factors, and output the quantization factors to a file.
4. Finally, the user can call the [save\_model](./en/api/save_model.md) API to save the quantized model, including a model file that can be used for accuracy simulation in the ONNX Runtime environment of the ONNX execution framework and a model file that can be deployed on the AI processor.
   - Accuracy simulation model file: an ONNX-format model file whose name contains **fake\_quant**. It can be used for accuracy simulation in the ONNX Runtime environment.

     The **fake\_quant** model is mainly used to verify the accuracy of the quantized model and can run in the ONNX Runtime environment. During the forward inference computation, the input data and weights of the convolutional layers and other layers are quantized and dequantized in the **fake\_quant** model to simulate the quantized computation results, thereby quickly verifying the accuracy of the quantized model. As shown in the following figure, taking INT8 quantization as an example, the data among the Quant layer, the Conv2d convolutional layer, and the DeQuant layer are all of the float32 data type. The Quant layer quantizes the data to INT8 and then dequantizes it back to float32, and the weights are also quantized to INT8 and then dequantized back to float32. The actual computation of the convolutional layer is based on the float32 data type. This model is used to verify the accuracy of the quantized model in the ONNX Runtime environment and cannot be used by the ATC tool to convert to an om model.

     **Figure 2**  fake\_quant model<a name="fig830317361367"></a>
     ![](./en/figures/fake_quant_module.png "fake_quant model")

   - Deployment model file: an ONNX-format model file whose name contains **deploy**. After conversion by the ATC conversion tool, it can be deployed to the AI processor.

     Taking INT8 quantization as an example, the deploy model has already converted weights and other data to INT8 and INT32 types, so it cannot perform inference computation in the ONNX Runtime environment. As shown in the following figure, the AscendQuant layer of the deploy model quantizes the float32 input data to INT8 as the input of the convolution layer. The weights are also computed using the INT8 data type. In the deploy model, the computation of the convolution layer is based on INT8 and INT32 data types, and the output is INT32 data, which is converted to float32 by the AscendDeQuant layer and then passed to the next network layer.

     **Figure 3**  deploy model<a name="fig32095209306"></a>
     ![](./en/figures/deploy_module.png "deploy model")

##### Call Example

> [!NOTE] Note
>
>- In the following examples, steps marked "to be supplemented by the user" require you to supplement the processing based on your own model and dataset. The code in the examples is for reference only.
>- For the parts of the following examples that call AMCT, adjust the function input parameters according to the actual situation.

1. Import the AMCT package, and set the log level through the environment variables in [Tool Building](./en/build.md).

   ```python
   import amct_pytorch as amct
   ```

2. <a name="li13259141318410"></a>(Optional, to be supplemented by the user) Verify the inference script and environment in the original PyTorch environment.

   It is recommended to use the original model to be quantized and the test set to perform inference in the PyTorch environment, so as to verify whether the environment and inference script work properly.

   It is recommended to perform this step to ensure that the original model can complete inference with normal accuracy. When performing this step, you can use a subset of the test set to reduce the running time.

   ```python
   user_do_inference_torch(ori_model, test_data, test_iterations)
   ```

3. Call AMCT to quantize the model.
   1. Generate the quantization configuration.

      ```python
      config_file = './tmp/config.json'
      skip_layers = []
      batch_num = 1
      amct.create_quant_config(config_file=config_file,
                model=ori_model,
                               input_data=ori_model_input_data,
                skip_layers=skip_layers,
                batch_num=batch_num)
      ```

   2. Modify the graph by inserting data quantization, weight quantization, and other related operators into the graph to compute quantization-related parameters.

      ```python
      record_file = './tmp/record.txt'
      modified_onnx_model = './tmp/modified_model.onnx'
      calibration_model = amct.quantize_model(config_file=config_file,
                     modfied_onnx_file=modified_onnx_model,
                     record_file=record_file,
                                              model=ori_model,
                                              input_data=ori_model_input_data)
      ```

   3. (To be supplemented by the user) Based on the PyTorch environment, perform model inference on the calibration set (**calibration\_data**) using the modified model (**calibration\_model**) to generate quantization factors.

      When performing this step, note the following two points:

      1. The calibration set and its preprocessing data must match the model to ensure quantization accuracy.
      2. The number of forward inference passes is **batch\_num**. If the number is insufficient, the calibration operators will not output the quantization factors to the record file, causing subsequent record reading verification to fail.

      ```python
      user_do_inference_torch(calibration_model, calibration_data, batch_num)
      ```

   4. Save the model.

      Based on the quantization factors, call the [save\_model](./en/api/save_model.md) API to insert operators such as AscendQuant and AscendDequant, and save the result as a quantized model.

      ```python
      quant_model_path = './results/user_model'
      amct.save_model(modfied_onnx_file=modified_onnx_model,
                      record_file=record_file,
                      save_path=quant_model_path)
      ```

4. (Optional, to be supplemented by the user) Based on the ONNX Runtime environment, perform inference on the test set (**test\_data**) using the quantized model (**quant\_model**) to test the accuracy of the quantized simulation model.

   Compare the accuracy of the quantized simulation model with the original accuracy in [2](#li13259141318410) to observe the impact of quantization on accuracy.

   ```python
   quant_model = './results/user_model_fake_quant_model.onnx'
   user_do_inference_onnx(quant_model, test_data, test_iterations)
   ```

##### Manual Tuning

If the accuracy of the post-training quantization feature does not meet requirements, you can try to manually adjust the parameters in the **config.json** file. This section provides the adjustment principles and parameter explanations.

Quantization is performed using the default configuration in the **config.json** file generated by the [create\_quant\_config](./en/api/create_quant_config.md) API. If the inference accuracy after quantization does not meet requirements, you can adjust the quantization configuration and repeat quantization until the accuracy meets requirements. This section describes the manual tuning process in detail. The objects to be adjusted are the parameters in the post-training quantization configuration file **config.json**, which mainly involve three stages:

1. Adjust the amount of data used for calibration.
2. Skip quantization for certain layers.
3. Adjust the quantization algorithm and parameters.

The specific steps are as follows:

1. Perform quantization using the default configuration generated by the [create\_quant\_config](./en/api/create_quant_config.md) API. If the accuracy meets requirements, tuning ends; otherwise, proceed to the next step.

2. <a name="li7823183151113"></a>Manually modify **batch\_num** to adjust the amount of data used for calibration.

   **batch\_num** controls the number of batches of data used for quantization, and can be adjusted based on the batch size and the number of images required for quantization. In general:

   The larger **batch\_num** is, the more data samples are used in the quantization process, and the smaller the accuracy loss after quantization. However, excessive data does not improve accuracy; instead, it consumes more memory, reduces quantization speed, and may cause insufficient memory, video memory, and thread resources. Therefore, it is recommended to set batch\_num\*batch\_size to 16 or 32 (**batch\_size** indicates the number of images used per batch).

3. If the model accuracy meets the requirement after quantization is performed using the quantization configuration in [2](#li7823183151113), the tuning ends. Otherwise, proceed to the next step.

4. <a name="li1582383101111"></a>Manually modify **quant\_enable** to skip quantization of certain layers.

   **quant\_enable** specifies whether a layer is quantized. When the value is **true**, the layer is quantized; when the value is **false**, the layer is not quantized. Deleting the configuration of the layer also skips quantization of that layer.

   When the accuracy of the entire network does not meet the requirement, you need to identify the quantization-sensitive layers in the network (layers whose error increases significantly after quantization), and then cancel quantization of these layers. There are two methods for identifying quantization-sensitive layers:

   1. Based on the network model structure, the first layer, the last layer, and layers with a small number of parameters generally suffer a significant accuracy drop after quantization.
   2. Use an accuracy comparison tool to compare the output error of the original model and the quantized model layer by layer (for example, using cosine similarity as the criterion, which requires a similarity above 0.99), find the layers with large errors, and roll back these layers first.

5. If the model accuracy meets the requirement after quantization is performed using the quantization configuration in [4](#li1582383101111), the tuning ends. Otherwise, proceed to the next step.

6. <a name="li6823133201119"></a>Manually modify **activation\_quant\_params** and **weight\_quant\_params** to adjust the quantization algorithm and parameters:

   For the meaning of the algorithm parameters, see [Post-Training Quantization Configuration Parameters](./en/context/ptq_config_param.md). For the algorithm description, see [Algorithm Introduction](./en/algorithm_brief.md).

7. If the model accuracy meets the requirement after quantization is performed using the quantization configuration in [6](#li6823133201119), the tuning ends. Otherwise, it indicates that quantization has a significant impact on accuracy and quantization cannot be performed. Remove the quantization configuration.

**Figure 4**  Parameter tuning process<a name="zh-cn_topic_0240188709_fig10376759162410"></a>
![](./en/figures/adjustprocess.png "Parameter tuning process")

### Quantization-Aware Training

#### Quantization Flow

The API call flow of quantization-aware training is shown in the following figure. In this flow, the training environment uses the CPU or NPU environment of the PyTorch framework. Based on the inference script of this open-source framework, AMCT APIs are called to complete model compression. The compressed deployment model must be converted into an offline model adapted to the AI processor using the ATC tool before inference can be performed on the AI processor. For the quantization algorithm used in quantization-aware training, see [Algorithm Introduction](./en/algorithm_brief.md).

![](./en/figures/qat_interface.png "API call flow-0")

The blue parts are implemented by the user, and the gray parts are implemented by calling the APIs provided by AMCT:

1. The user first constructs the original PyTorch model and calls the [create\_quant\_retrain\_config](./en/api/create_quant_retrain_config.md) API to generate the quantization configuration file.
2. Call the [create\_quant\_retrain\_model](./en/api/create_quant_retrain_model.md) API to modify the original model. Operators related to data quantization and weight quantization are inserted into the modified model to calculate quantization-related parameters.
3. Train the modified model. If the training is not interrupted, perform inference on the trained model. During inference, the quantization factors are written to the record file.

   If the training is interrupted, you can call the [restore\_quant\_retrain\_model](./en/api/restore_quant_retrain_model.md) API again based on the saved **pth** model parameters and the quantization configuration file to output the modified retrain network, continue the quantization-aware training, and then perform inference.

4. Call the [save\_quant\_retrain\_model](./en/api/save_quant_retrain_model.md) API to insert quantization operators such as AscendQuant and AscendDequant, and save the quantized model.

#### Call Example

> [!NOTE] Note
>
>1. For training based on the PyTorch environment, currently only multi-device training in distributed mode (that is, DistributedDataParallel mode) is supported. Multi-device training in DataParallel mode is not supported, and training in DP mode will report an error.
>2. For the parts that call AMCT, the function input parameters can be adjusted as needed. Quantization-aware training is based on the user's training process. Ensure that you already have a training script based on the PyTorch environment and that the accuracy after training is normal.
>3. When using the quantization-aware training feature of AMCT, if the training process hangs, check whether other ONNX Runtime programs are running on the current server (you can use the **top** command to view all processes on the server). If so, pause the other ONNX Runtime programs first and then re-execute quantization-aware training.
>4. When performing quantization by referring to this chapter, if the model contains PyTorch custom operators, the ONNX model may fail to be exported, resulting in quantization failure. The specific error message is as follows: 'Model cannot be quantized for it cannot be export to onnx!'. In this case, refer to [Quantization-Aware Training in Single-Operator Mode](#quantization-aware-training-in-single-operator-mode) to perform quantization in single-operator mode.
>5. For the steps marked with "to be supplemented by the user" in the following examples, the user needs to supplement the processing based on their own model and dataset. The examples only provide sample code.

1. Import the AMCT package and set the log level through the environment variables in [Tool Building](./en/build.md).

   ```python
   import amct_pytorch as amct
   ```

2. <a name="li341817311449"></a>(Optional, to be supplemented by the user) It is recommended to use the original model to be quantized and the test set to perform inference in the PyTorch environment to verify that the environment and inference script work properly.

   It is recommended to perform this step to ensure that the original model can complete inference with normal accuracy. When performing this step, you can use a subset of the test set to reduce the running time.

   ```python
   ori_model.load()
   # Test the model.
   user_test_model(ori_model, test_data, test_iterations)
   ```

3. Call AMCT to execute the quantization process.
   1. Generate the quantization configuration.

      Before implementing this step, restore the trained parameters, such as **ori\_model.load\(\)** in the previous step.

      ```python
      config_file = './tmp/config.json'
      simple_cfg = './retrain.cfg'
      amct.create_quant_retrain_config(config_file=config_file,
                                       model=ori_model,
                                       input_data=ori_model_input_data,
                                       config_defination=simple_cfg)
      ```

   2. <a name="li4981151425915"></a>Modify the model.

      Insert data quantization, weight quantization, and other related operators into the model **ori\_model** to calculate quantization-related parameters, and then save it as a new training model (**retrain\_model**).

      ```python
      record_file = './tmp/record.txt'
      quant_retrain_model = amct.create_quant_retrain_model(config_file=config_file,
                              model=ori_model,
                              record_file=record_file,
                              input_data=ori_model_input_data)
      ```

   3. (To be supplemented by the user) Use the modified graph to create backward gradients, perform training on the training set, and train the quantization factors.
      1. Use the modified graph to create the backward gradient. This step must be performed after [3.b](#li4981151425915).

         ```python
         optimizer = user_create_optimizer(quant_retrain_model)
         ```

      2. Restore the model from the trained checkpoint and train the model.

         Note: Restore the model parameters from the trained checkpoint before training. The parameters saved during training must include the quantization factors: the quantization factors are generated after the first **batch\_num** training iterations. If the number of training iterations is fewer than **batch\_num**, the process fails.

         ```python
         quant_pth = './ckpt/user_model'
         user_train_model(optimizer, quant_retrain_model, train_data)
         ```

      3. After training is complete, perform inference to compute and save the quantization factors.

         ```python
         user_infer_graph(quant_retrain_model)
         ```

   4. Save the quantized model.

      Based on the quantization factors and the user-retrained model, call the [save\_quant\_retrain\_model](./en/api/save_quant_retrain_model.md) API to insert operators such as AscendQuant and AscendDequant, and save the result as the quantized model.

      ```python
      quant_model_path = './result/user_model'
      amct.save_quant_retrain_model(config_file=config_file,
                                    model=quant_retrain_model,
                                    record_file=record_file,
                                    save_path=quant_model_path,
                                    input_data=ori_model_input_data)
      ```

      Note: If this API is called to save the model in every retraining iteration, retraining may become abnormal. It is recommended to call this API to save the model only after retraining is complete.

4. (Optional, to be supplemented by the user) Based on the ONNX Runtime environment, use the quantized model (**quant\_model**) to perform inference on the test set (**test\_data**) and test the accuracy of the quantized simulation model.

   By comparing the accuracy of the quantized simulation model with the original accuracy in [2](#li341817311449), you can observe the impact of quantization on accuracy.

   ```python
   quant_model = './results/user_model_fake_quant_model.onnx'
   user_do_inference_onnx(quant_model, test_data, test_iterations)
   ```

**If the training process is interrupted and data needs to be restored from the ckpt to continue training, the call flow is as follows:**

1. Import the AMCT package and set the log level through the environment variables in [Tool Building](./en/build.md).

   ```python
   import amct_pytorch as amct
   ```

2. Prepare the original model.

   ```python
   ori_model= user_create_model()
   ```

3. Call AMCT to resume the quantization training process.
   1. <a name="li413111408612"></a>Modify the model, insert quantization-related operators into the model **ori\_model**, and save it as a new training model (**retrain\_model**).

      ```python
      config_file = './tmp/config.json'
      simple_cfg = './retrain.cfg'
      record_file = './tmp/record.txt'
      quant_pth_file = './ckpt/user_model_newest.ckpt'
      quant_retrain_model = amct.restore_quant_retrain_model(config_file=config_file,
                               model=ori_model,
                               record_file=record_file,
                               input_data=ori_model_input_data,
                               pth_file=quant_pth_file)
      ```

   2. (To be supplemented by the user) Use the modified graph to create backward gradients, train on the training set, and train the quantization factors.
      1. Use the modified graph to create backward gradients. This step must be performed after [3.a](#li413111408612).

         ```python
         optimizer = user_create_optimizer(retrain_model)
         ```

      2. Restore the model from the trained checkpoint and train the model.

         Note: Restore the model parameters from the trained checkpoint before training. The parameters saved during training must include the quantization factors: quantization factors are generated after the first **batch\_num** training runs. If the number of training runs is fewer than **batch\_num**, the process fails.

         ```python
         user_train_model(optimizer, retrain_model, train_data)
         ```

      3. After training is complete, perform inference to compute and save the quantization factors.

         ```python
         user_infer_graph(train_graph, retrain_ops[-1].output_tensor)
         ```

   3. Save the model.

      ```python
      quant_model_path = './result/user_model'
      amct.save_quant_retrain_model(config_file=config_file,
                                    model=ori_model,
                                    record_file=record_file,
                                    save_path=quant_model_path,
                                    input_data=ori_model_input_data)
      ```

4. (Optional, to be supplemented by the user) Based on the ONNX Runtime environment, perform inference on the test set (**test\_data**) using the quantized model (**quant\_model**) to test the accuracy of the quantized simulation model.

   Compare the accuracy of the quantized simulation model with the original accuracy in [2](#li341817311449) to observe the impact of quantization on model accuracy.

   ```python
   quant_model = './results/user_model_fake_quant_model.onnx'
   user_do_inference_onnx(quant_model, test_data, test_iterations)
   ```

#### Manual Tuning

If the accuracy after performing the quantization-aware training feature does not meet the requirements, you can try to manually adjust the parameters in the **config.json** file. This section provides the adjustment principles and parameter explanations.

Quantization is performed using the default configuration in the **config.json** file generated by the [create\_quant\_retrain\_config](./en/api/create_quant_retrain_config.md) API. If the inference accuracy after quantization does not meet the requirements, adjust the parameters in the quantization configuration file by following the steps below.

1. Perform quantization based on the default configuration generated by the [create\_quant\_retrain\_config](./en/api/create_quant_retrain_config.md) API. If the accuracy meets the requirements, parameter tuning ends; otherwise, proceed to the next step.

2. In the INT8 quantization scenario, you can cancel quantization for some quantization layers, that is, change their **retrain\_enable** parameter to **false**. Generally, the first and last layers of a model have a greater impact on the inference result, so it is recommended to cancel quantization for the first and last layers first. If you have recommended values for the **clip\_max** and **clip\_min** parameters, you can modify the quantization configuration file as follows:

   ```json
   {
       "version":1,
       "batch_num":1,
       "layername1":{
           "retrain_enable":true,
           "retrain_data_config":{
               "algo":"ulq_quantize",
               "clip_max":3.0,
               "clip_min":-3.0
           },
           "retrain_weight_config":{
               "algo":"arq_retrain",
               "channel_wise":true
           }
       },
       "layername2":{
           "retrain_enable":true,
           "retrain_data_config":{
               "algo":"ulq_quantize",
               "clip_max":3.0,
               "clip_min":-3.0
           },
           "retrain_weight_config":{
               "algo":"arq_retrain",
               "channel_wise":true
           }
       }
   }
   ```

3. After the configuration is complete, if the accuracy meets the requirements, parameter tuning ends; otherwise, it indicates that quantization-aware training has a significant impact on accuracy and cannot be performed. Remove the quantization-aware training configuration.

For parameter descriptions in the configuration file, see [Quantization-Aware Training Configuration Parameter Description](./en/context/qat_config_param.md).

### Quantization-Aware Training in Single-Operator Mode

#### Feature Description

When performing basic quantization by referring to the [Quantization-Aware Training](#quantization-aware-training) section, the internal quantization processing logic needs to convert the original model into an ONNX model and modify the graph based on the ONNX model. In this case, if the model contains PyTorch custom operators, the ONNX model may fail to be exported, resulting in quantization failure.

The quantization-aware training feature in single-operator mode provides custom QAT operators generated by converting PyTorch native operators. Based on these operators, quantization factors are retrained, and the quantization factors are stored in the QAT single operator as operator parameters. **There is no need to export an ONNX model**, which avoids the operator export exception problem in the quantization-aware training solution described above. After training is completed, the mapping between QAT operators and ONNX native operators is established through the **torch.onnx.export** mechanism, and the parameters obtained by the QAT operators in the PyTorch model are passed to the ONNX native quantization operators to complete model export. A simple schematic diagram is shown below.

![](./en/figures/singleop_qat.png "Quantization-aware training in single-operator mode")

The implementation principle is shown in the following figure (using the torch.nn.Conv2d operator as an example):

The ARQ weight quantization and ULQ data quantization operators are inserted into the torch native operator torch.nn.Conv2d to retrain the quantization factors. The quantization factors are stored in the Conv2dQAT single operator as operator parameters. Finally, the torch.onnx.export API is called to export the custom QAT operators to the ONNX native quantization operators QuantizeLinear and DequantizeLinear through the mapping relationship. A model with these two quantization operators is also called a QDQ (Quantize and DeQuantize, QDQ for short) ONNX model. For details about the QDQ model format, click [here](https://onnxruntime.ai/docs/performance/quantization.html).

A QDQ ONNX model cannot be directly converted into an offline model adapted to the AI processor through the ATC tool. You need to use the [QAT model adaptation to CANN model](<https://hiascend.com/document/redirect/CannCommunityToolAmct_ConvertQAT_Model>) feature to adapt the model into a CANN model, and then use the ATC tool for the subsequent conversion.

![](./en/figures/pytorch_singleop_v1.png)

This feature supports two usage methods: training from scratch and fine-tune.

- Training from scratch: **Directly build the graph using QAT operators** and train from scratch. You can call the direct construction APIs provided by [single-operator mode](./en/api/README.md) in the model building script to construct QAT operators, and use these operators for model building.
- Fine-tune: **Replace the operators to be quantized on the basis of an existing network**, which is more commonly used than training from scratch. If you have completed model construction, you can call the native-operator-based construction API provided by [single-operator mode](./en/api/README.md) to construct QAT operators. Then, refer to the operator replacement scheme in the following example to replace the operators to be quantized in the network model.

The QAT operator specifications are shown in the following table:

<a name="table69463311346"></a>

<table><thead align="left"><tr id="row179476303418"><th class="cellrowborder" valign="top" width="16.05839416058394%" id="mcps1.2.5.1.1"><p id="p629775412918"><a name="p629775412918"></a><a name="p629775412918"></a>Operator Type to Be Quantized</p></th>
<th class="cellrowborder" valign="top" width="15.418458154184583%" id="mcps1.2.5.1.2"><p id="p59476317345"><a name="p59476317345"></a><a name="p59476317345"></a>Operator Type After Replacement</p></th>
<th class="cellrowborder" valign="top" width="50.34496550344964%" id="mcps1.2.5.1.3"><p id="p89470353414"><a name="p89470353414"></a><a name="p89470353414"></a>Restriction</p></th>
<th class="cellrowborder" valign="top" width="18.17818218178182%" id="mcps1.2.5.1.4"><p id="p59471732344"><a name="p59471732344"></a><a name="p59471732344"></a>Remarks</p></th>
</tr>
</thead>
<tbody><tr id="row2094723143420"><td class="cellrowborder" valign="top" width="16.05839416058394%" headers="mcps1.2.5.1.1 "><p id="p1929725415291"><a name="p1929725415291"></a><a name="p1929725415291"></a>torch.nn.Conv2d</p></td>
<td class="cellrowborder" valign="top" width="15.418458154184583%" headers="mcps1.2.5.1.2 "><p id="p59474320349"><a name="p59474320349"></a><a name="p59474320349"></a>Conv2dQAT</p></td>
<td class="cellrowborder" valign="top" width="50.34496550344964%" headers="mcps1.2.5.1.3 "><a name="ul143605432616"></a><a name="ul143605432616"></a><ul id="ul143605432616"><li>padding_mode is zeros.</li><li>Only 4-dimensional input is supported.</li></ul></td>
<td class="cellrowborder" rowspan="4" valign="top" width="18.17818218178182%" headers="mcps1.2.5.1.4 "><p id="p713110161816"><a name="p713110161816"></a><a name="p713110161816"></a>Reused layers (sharing weight and bias parameters) do not support quantization.</p></td>
</tr>
<tr id="row9947536345"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p4297185402911"><a name="p4297185402911"></a><a name="p4297185402911"></a>torch.nn.ConvTranspose2d</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p17947113143414"><a name="p17947113143414"></a><a name="p17947113143414"></a>ConvTranspose2dQAT</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><a name="ul154411347463"></a><a name="ul154411347463"></a><ul id="ul154411347463"><li>padding_mode is zeros.</li><li>Only 4-dimensional input is supported.</li></ul></td>
</tr>
<tr id="row1494715383411"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p5344050319"><a name="p5344050319"></a><a name="p5344050319"></a>torch.nn.Conv3d</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1894713320342"><a name="p1894713320342"></a><a name="p1894713320342"></a>Conv3dQAT</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><a name="ul72031530619"></a><a name="ul72031530619"></a><ul id="ul72031530619"><li>padding_mode is zeros.</li><li>dilation_d is 1.</li><li>Only 5-dimensional input is supported.</li></ul></td>
</tr>
<tr id="row594718313414"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p715801183117"><a name="p715801183117"></a><a name="p715801183117"></a>torch.nn.Linear</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p49474316347"><a name="p49474316347"></a><a name="p49474316347"></a>LinearQAT</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p199471231348"><a name="p199471231348"></a><a name="p199471231348"></a>Channel-wise quantization is not supported.</p></td>
</tr>
</tbody>
</table>

#### Call Example

Code example for the training-from-scratch method:

```python
import torch
import torch.nn as nn

import amct_pytorch as amct
from amct_onnx.convert_model import convert_qat_model
from amct_pytorch.nn.module.quantization.conv2d import Conv2dQAT
from amct_pytorch.nn.module.quantization.linear import LinearQAT


# QAT operator quantization configuration items.
config = {
    "retrain_data_config": {
        "dst_type": "INT8",
        "batch_num": 10,
        "fixed_min": False,
        "clip_min": None,
        "clip_max": None
    },
    "retrain_weight_config": {
        "dst_type": "INT8",
        "weights_retrain_algo": "arq_retrain",
        "channel_wise": False
    }
}

# Construct the Lenet network using QAT single operators.
net = torch.nn.Sequential(
    Conv2dQAT(1, 6, kernel_size=5, padding=2, config=config), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    Conv2dQAT(6, 16, kernel_size=5, config=config), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    LinearQAT(16 * 5 * 5, 120, config=config), nn.Sigmoid(),
    LinearQAT(120, 84, config=config), nn.Sigmoid(),
    LinearQAT(84, 10, config=config)
)

# Train.
train(net, train_data, test_data)
# Export the intermediate model.
torch.onnx.export(model, data, 'inter_model.onnx')

# Export the fake quant model and the deploy model.
# Generate lenet_fake_quant_model.onnx, a model that can be used for accuracy simulation in the ONNX Runtime execution framework.
# Generate lenet_deploy_model.onnx, a model file that can be deployed on AI processors.
convert_qat_model('inter_model.onnx', './outputs/lenet')

# Use the fake quant model for accuracy simulation.
validate_onnx('./outputs/lenet_fake_quant_model.onnx', val_data)
```

Code example for the fine-tune method:

```python
import torch
from torchvision.models.resnet import resnet101

import amct_pytorch as amct
from amct_onnx.convert_model import convert_qat_model
from amct_pytorch.nn.module.quantization.conv2d import Conv2dQAT

model = resnet101(pretrained=True)
# QAT operator quantization configuration items.
config = {
    "retrain_data_config": {
        "dst_type": "INT8",
        "batch_num": 10,
        "fixed_min": False,
        "clip_min": None,
        "clip_max": None
    },
    "retrain_weight_config": {
        "dst_type": "INT8",
        "weights_retrain_algo": "arq_retrain",
        "channel_wise": True
    }
}

def _set_module(model, submodule_key, module):
    # Replace the native operators in the model with QAT operators.
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

for name, module in model.named_modules():
    # Traverse the nodes in the original graph and replace the torch native operators of type Conv2d with custom QAT single operators.
    if isinstance(module, torch.nn.Conv2d):
        qat_module = Conv2dQAT.from_float(
            module, config=config)
        _set_module(model, name, qat_module)

# Training process.
train_and_val(model, train_data, test_data)
# Export the intermediate model.
torch.onnx.export(model, data, 'inter_model.onnx')

# Export the fake quant model and the deploy model.
# Generate resnet101_fake_quant_model.onnx, a model on which accuracy simulation can be performed in the ONNX Runtime execution framework.
# Generate resnet101_deploy_model.onnx, a model file that can be deployed on AI processors.
convert_qat_model('inter_model.onnx', './outputs/resnet101')

# Use the fake quant model for accuracy simulation.
validate_onnx('./outputs/resnet101_fake_quant_model.onnx', val_data)
```

### Quantization Data Balancing Preprocessing

In some scenarios with uneven data distribution, per-tensor quantization of data yields large errors due to the influence of outliers, whereas per-channel quantization can achieve lower errors. However, because the current hardware does not support per-channel quantization of data and only supports per-channel quantization of weights, AMCT designs a method for this context, which is described in detail in this section.

The balancing factors are calculated through the data balancing preprocessing API, and the model data and weights are mathematically equivalently converted to balance the distribution of model data and weights, thereby shifting part of the quantization difficulty from data to weights and reducing the quantization error. For the layers and constraints supported by this feature, see [Layers and Constraints Supported by Quantization Balancing Preprocessing](./en/api/create_quant_config.md#layers-and-constraints-supported-by-balanced-quantization-preprocessing).

#### Quantization Flow

The call flow of the balancing preprocessing API is shown in the following figure.

![](./en/figures/dmq_balancer.png "Balancing preprocessing API call flow")

The blue part is implemented by the user, and the gray part is implemented by calling the APIs provided by AMCT. The tool usage is divided into the following scenarios:

1. The user first constructs the original PyTorch model and sets the **dmq** parameters in the simple configuration file **_dmp\_quant.cfg_** (for the configuration parameters of the simple configuration file, see **DMQBalancer** in [Simple Configuration File for Post-Training Quantization](./en/context/ptq_config.md)), and then passes the configuration file to [create\_quant\_config](./en/api/create_quant_config.md).
2. <a name="li16458161113715"></a>Based on the PyTorch model and the quantization configuration file, the [quantize\_preprocess](./en/api/quantize_preprocess.md) API can be called to optimize the original PyTorch model. The optimized PyTorch model contains the balanced quantization algorithm.
3. Use the calibration set to perform **one** forward inference in the PyTorch environment to generate the balanced quantization factors, and output the balanced quantization factors to a file.
4. Based on the PyTorch model, the quantization configuration file, and the record file, the [quantize\_model](./en/api/quantize_model.md) API can be called to optimize the original PyTorch model again. The optimized PyTorch model contains the quantization algorithm.
5. Use the calibration set to perform forward inference in the PyTorch environment to generate quantization factors, and output the quantization factors to a file.
6. Finally, the user can call the [save\_model](./en/api/save_model.md) API to save the quantized model, including a model file that can be used for accuracy simulation in the ONNX Runtime environment and a model file that can be deployed on the AI processor.

#### Call Example

> [!NOTE] Note
>
>- In the following example, the steps marked "to be supplemented by the user" need to be supplemented by the user based on their own model and dataset. The example provides only sample code.
>- For the parts of the following example that call AMCT, adjust the function input parameters according to the actual situation.

1. Import the AMCT package, and set the log level through the environment variables in [Tool Building](./en/build.md).

   ```python
   import amct_pytorch as amct
   ```

2. (Optional, to be supplemented by the user) Verify the inference script and environment in the original PyTorch environment.

   It is recommended to use the original model to be quantized and the test set to perform inference in the PyTorch environment to verify that the environment and the inference script work properly.

   It is recommended to perform this step. Ensure that the original model can complete inference with normal accuracy. When performing this step, you can use part of the test set to reduce the running time.

   ```python
   user_do_inference_torch(ori_model, test_data, test_iterations)
   ```

3. Call AMCT to quantize the model.
   1. Generate the quantization configuration.

      The user sets the **dmq** parameters in the simple configuration file **_dmp\_quant.cfg_** and passes the configuration file to [create\_quant\_config](./en/api/create_quant_config.md) through the **config\_defination** parameter.

      ```python<ins></ins>
      config_defination = os.path.join(PATH, 'dmp_quant.cfg')
      config_file = './tmp/config.json'
      skip_layers = []
      batch_num = 1
      amct.create_quant_config(config_file=config_file,
                       model=ori_model,
                       input_data=ori_model_input_data,
                       skip_layers=skip_layers,
                       batch_num=batch_num,
                       config_defination=config_defination)
      ```

   2. Modify the graph and insert balanced quantization operators into the graph to calculate the balanced quantization factors.

      ```python
      record_file = './tmp/record.txt'
      calibration_model = amct.quantize_preprocess(config_file=config_file,
                                record_file=record_file,
                                model=ori_model,
                                input_data=ori_model_input_data)
      ```

   3. (To be supplemented by the user) Based on the PyTorch environment, perform model inference on the calibration set (**calibration\_data**) using the modified model (**calibration\_model**) to find the balanced quantization factors.

      When performing this step, note the following two points:

      1. The calibration set and its preprocessing data must match the model to ensure quantization accuracy.
      2. The number of forward inference runs is 1. If it exceeds 1, the balanced quantization factors in the record are recorded once per inference run, causing the subsequent process to fail.

      ```python
      user_do_inference_torch(calibration_model, calibration_data, test_iterations=1)
      ```

   4. Modify the graph and insert operators related to data quantization and weight quantization into the graph to calculate quantization-related parameters.

      ```python
      modified_onnx_model = './tmp/modified_model.onnx'
      calibration_model = amct.quantize_model(config_file=config_file,
                            modfied_onnx_file=modified_onnx_model,
                            record_file=record_file,
                            model=ori_model,
                            input_data=ori_model_input_data)
      ```

   5. (To be supplemented by the user) Based on the PyTorch environment, perform model inference on the calibration set (**calibration\_data**) using the modified model (**calibration\_model**) to generate quantization factors.

      When performing this step, note the following two points:

      1. The calibration set and its preprocessing data must match the model to ensure quantization accuracy.
      2. The number of forward inference runs is **batch\_num**. If the number is insufficient, the subsequent process will fail.

      ```python
      user_do_inference_torch(calibration_model, calibration_data, batch_num)
      ```

   6. Save the model.

      Based on the quantization factors, call the [save\_model](./en/api/save_model.md) API to insert operators such as AscendQuant and AscendDequant, and save the result as a quantized model.

      ```python
      quant_model_path = './results/user_model'
      amct.save_model(modfied_onnx_file=modified_onnx_model,
                             record_file=record_file,
                             save_path=quant_model_path)
      ```

4. (Optional, to be supplemented by the user) In an ONNX Runtime environment, use the quantized model (**quant\_model**) to perform inference on the test set (**test\_data**) and test the accuracy of the quantized simulation model.

   By comparing the accuracy of the quantized simulation model with the original accuracy in [2](#li16458161113715), you can observe the impact of quantization on accuracy.

   ```python
   quant_model = './results/user_model_fake_quant_model.onnx'
   user_do_inference_onnx(quant_model, test_data, test_iterations)
   ```

### KV Cache Quantization

KV Cache (Key-Value Cache) is a caching technique that reuses computation results by storing key-value pairs, thereby improving performance and reducing memory consumption. In large-scale training and inference, as **batch\_size** and **sequence\_length** continue to grow, the memory overhead occupied by the KV Cache also increases rapidly, and may even exceed that of the model itself. Applying INT8 quantization to the KV Cache can significantly reduce memory consumption during model execution and lower model deployment costs.

**Layers supported by KV Cache quantization: torch.nn.Linear, with the original data types being float32 and float16**.

#### Quantization Flow

The KV Cache quantization API call flow is shown in the following figure.

![](./en/figures/kv_cache_quant.png "KV Cache quantization call flow")

The blue parts are implemented by the user, and the gray parts are implemented by calling the APIs provided by AMCT. The brief flow is as follows:

1. The user first constructs the original PyTorch model, and then uses [create\_quant\_cali\_config](./en/api/create_quant_cali_config.md) to generate the quantization configuration file.
2. Based on the PyTorch model and the quantization configuration file, the user locates the Linear operators to be quantized in the model, and then calls the [create\_quant\_cali\_model](./en/api/create_quant_cali_model.md) API to modify the original PyTorch model, adding IFMR/HFMG quantization operators after the Linear output.
3. The user performs forward inference in the PyTorch environment using the calibration set, calls the IFMR/HFMG quantization algorithm to calibrate the output according to the quantization algorithm configuration, and outputs the results to the quantization factor file in the corresponding format.

**If the user needs to support more operator types, or has customized other operations, the [QuantCalibrationOp](./en/api/QuantCalibrationOp.md) API can be used to construct the graph, perform quantization calibration, and output the quantization factor file.**

#### Call Example

> [!NOTE] Note
>
>- In the following examples, steps marked with "to be supplemented by the user" require you to supplement the processing based on your own model and dataset. The code in the examples is for reference only.
>- For the parts that call AMCT in the following examples, adjust the function input parameters based on the actual situation.

1. Import the AMCT package and set the log level through the environment variables in [Tool Building](./en/build.md).

   ```python
   import amct_pytorch as amct
   ```

2. (Optional, to be supplemented by the user) Verify the inference script and environment in the original PyTorch environment.

   It is recommended to use the original model to be quantized and the test set to perform inference in the PyTorch environment, so as to verify that the environment and inference script work properly.

   It is recommended to perform this step. Ensure that the original model can complete inference with normal accuracy. When performing this step, you can use a subset of the test set to reduce the running time.

   ```python
   user_do_inference_torch(ori_model, test_data, test_iterations)
   ```

3. Call AMCT to quantize the model.
   1. Parse the quantization configuration file **quant.cfg** to generate the quantization configuration file.

      ```python
      config_file = './tmp/config.json'
      amct.create_quant_cali_config(config_file=config_file,
                        model=ori_model,
                        quant_layers=None,
                        config_defination="./configs/quant.cfg")
      ```

   2. Modify the graph and insert quantization-related operators into the graph to compute quantization-related parameters.

      Locate the Linear operator to be quantized in the model, insert the IFMR/HFMG quantization operator after the Linear output, and perform online calibration to compute the corresponding quantization factor, so that the output can be quantized to INT8 later.

      ```python
      record_file = './tmp/record.txt'
      calibration_model = amct.create_quant_cali_model(config_file=config_file,
                                 record_file=record_file,
                                 model=ori_model)
      ```

   3. (To be supplemented by the user) Based on the PyTorch environment, use the modified model (**calibration\_model**) to perform model inference on the calibration set (**calibration\_data**) to generate the quantization factor.

      When performing this step, note the following two points:

      1. The calibration set and its preprocessing data must match the model to ensure quantization accuracy.
      2. The number of forward inference runs is **batch\_num**. If the number is insufficient, the subsequent process will fail.

      ```python
      user_do_inference_torch(calibration_model, calibration_data, batch_num)
      ```

   4. Output the quantization factor file.

## Sparsity

AMCT provides a graph-based sparsity method for torch models. This method converts a torch model into an ONNX model to obtain the graph structure of the torch model, performs sparsity optimization based on the parsed graph structure, and finally generates an ONNX quantized model. Both the channel sparsity and the 4:2 structured sparsity described in this section are implemented based on this method.

### Channel Sparsity

#### Automatic Channel Sparsity

##### Feature Description

Currently, channel sparsity supports applying different sparsity rates to different layers. However, setting the sparsity rate layer by layer imposes a high usage threshold on users. For a given layer, it is difficult to determine how to select the sparsity rate (that is, **prune\_ratio** in the configuration). Manually attempted configurations require retraining, which is time-consuming. To address the above issues, the automatic channel sparsity search feature is introduced. It calculates the sparsity sensitivity (which affects accuracy) and sparsity benefit (which affects performance) of each channel based on the user model, and then the search strategy searches for the optimal layer-by-layer channel sparsity rate based on sparsity sensitivity and sparsity benefit to balance accuracy and performance.

- Sparsity sensitivity: Sparsity sensitivity is defined as the estimated impact on the accuracy of the entire network after the current channel is sparsified. A larger sparsity sensitivity indicates a greater accuracy loss of the entire network after the current channel is sparsified. By default, the algorithm calculates the channel sparsity sensitivity based on the Taylor expansion of loss\(w - w<sub>i</sub>\). User-defined calculation methods are supported. After pruning the i-th channel, the sparsity sensitivity is calculated as follows:

  $$

  v_i=loss(w)-loss(w-W_i)

  $$

  An approximate estimation method is used to perform Taylor expansion on loss\(w - w<sub>i</sub>\). Currently, only the first order is calculated considering the computational cost.

- Sparsity benefit: The sparsity benefit of the current channel is represented by bit complexity, which is the product of the quantized Flops and the computational bit width:

  $$

  bit\_complexity = Flops \times act\_bit \times wts\_bit

  $$

  Here, Flops is the floating-point computation amount, **act\_bit** is the data precision, and **wts\_bit** is the weight precision.

The automatic channel sparsity search process is shown in the following figure. For the supported layers and specifications, see [supported layers and specifications for channel sparsity](./en/api/create_prune_retrain_model.md#supported-layers-and-constraints-for-channel-sparsity).

![](./en/figures/autochannelprune.png "Automatic channel sparsity search process")

The processes are briefly described as follows:

1. Initialization: The initialization first parses the user model and the sparsity configuration (optional), analyzes the channel-sparsifiable layers in the network and their corresponding channel sparsity configurations (whether the user has specified a sparsity rate), and generates the search space for channel sparsity configuration generation: parses the user's target compression rate configuration.

   - Search space: layers that support channel sparsity but have no sparsity rate configured through **override\_layer\_configs** or **override\_layer\_types**.
   - Compression rate: defined as the ratio of the bit complexity of the original model to that of the sparsified model.

2. Sensitivity calculation: calculates the sparsity sensitivity of each channel. A built-in sensitivity calculation method based on loss estimation uses the first-order term of the Taylor expansion to estimate the change in network loss after pruning the channel. User-defined sensitivity calculation methods are supported.

3. Bit complexity calculation: calculates the bit complexity of each channel, which is regarded as the sparsity benefit of the channel.

4. Channel sparsity rate configuration search: by default, the [automatic channel sparsity search algorithm](./en/algorithm_brief.md) is used to search for the optimal channel sparsity rate configuration that meets the user-specified compression rate. User-defined solvers are supported.

   > [!NOTE] Note
   > The automatic channel sparsity search feature only generates a simple configuration file for channel sparsity. To obtain the final sparsified model, you still need to perform [Manual Sparsity](#manual-sparsity), passing the simple configuration file generated above as an input to channel sparsity.

##### Sparsity Process

The API call flow is shown in the following figure. The blue part is implemented by the user, and the gray part is implemented by calling the APIs provided by AMCT.

The user prepares the PyTorch model and the automatic channel sparsity search configuration file, calls [auto\_channel\_prune\_search](./en/api/auto_channel_prune_search.md), and performs automatic channel sparsity search based on the compression rate, the sparsity sensitivity of each channel, and the sparsity benefit to obtain a simple configuration file that can be used for channel sparsity. The sensitivity module and the **search\_alg** module can be customized by the user or use the default methods inside the API.

- The sensitivity module calculates the sparsity sensitivity of each channel.
- The **search\_alg** module implements the process of searching for sparse channels based on channel sensitivity and channel sparsity benefit.

![](./en/figures/autoprune.png "Call flow-1")

##### Call Example

This example demonstrates the process of using AMCT to perform automatic channel sparsity search. This process requires the user to pass in the PyTorch model and calibration data. The user can choose to customize the sensitivity module and the **search\_alg** module.

> [!NOTE] Note
>
>- For the steps marked with "to be supplemented by the user" in the following example, the user needs to supplement the processing based on their own model and dataset. The example provides only sample code.
>- For the parts that call AMCT in the following example, adjust the function input parameters based on the actual situation.

1. Import the AMCT package and set the log level through the environment variables in [Tool Building](./en/build.md).

   ```python
   import amct_pytorch as amct
   ```

2. (Optional, to be supplemented by the user) Implement the sensitivity module to obtain the sensitivity of each channel of each layer, providing data for the subsequent search algorithm. You can refer to the system default sensitivity module: _AMCT installation directory_

   **TaylorLossSensitivity** method in **amct\_pytorch/auto\_channel\_prune\_search.py**. The brief process is as follows:

   ```python
   from amct.common.auto_channel_prune.sensitivity_base import SensitivityBase

   class Sensitivity(SensitivityBase)
       def __init__(self)
           pass
       def setup_initialization(self, graph_tuple, input_data, test_iteration):
           # Perform necessary initialization.
           # graph_tuple (graph, graph_info)
           pass
       def get_sensitivity(self, search_records):
           # Obtain the sensitivity method, compute it, and write the result to the record.
           pass
   ```

3. (Optional, to be supplemented by the user) Implement the search algorithm **search\_alg** module. The user needs to implement the **channel\_prune\_search** callback API to search for sparse channels based on channel sensitivity and channel sparsity benefits. You can refer to the system default **search\_alg** module: the **GreedySearch** method in the **__AMCT installation directory__/amct\_pytorch/common/auto\_channel\_prune/search\_channel\_base.py** file.

   ```python
   from amct.common.auto_channel_prune.search_channel_base import SearchChannelBase

   class Search(SearchChannelBase)
       def __init__(self)
           # Initialize.
           pass

       def channel_prune_search(self, graph_info, search_records, prune_config):
           """
           Input:
           graph_info: dict, containing the channel count and bit complexity information of each operator in the graph, which can be used to calculate the compression rate.
           search_records: protobuf object, containing the sparsifiable layers to be searched.
           prune_config: a triple consisting of the target compression rate (float), the Ascend affinity optimization switch (bool), and the maximum sparsity rate of a single layer (float).
           Output:
           dict, where the key is the name of the sparsifiable layer to be searched and the value is a list of 0s and 1s indicating whether the corresponding channel should be sparsified
           """
           pass
   ```

4. (Optional, to be supplemented by the user) Obtain the PyTorch model and run inference in the environment to verify that the environment and inference script work properly. When performing this step, you can use a subset of the test set to reduce the runtime.

   ```python
   ori_model.load()
   # Test the model.
   user_test_model(ori_model, test_data, test_iterations)
   ```

5. (To be supplemented by the user) Construct the calibration data.

   ```python
   input_data = []
   for _ in range(test_iteration):
       input_data.append(user_load_feed_dict())
   ```

6. Call AMCT to perform automatic channel sparsity search.

   ```python
   output_prune_cfg = './prune.cfg'
   amct.auto_channel_prune_search(
       model=model,
       config=cfg_file,
       input_data=input_data,
       output_cfg=output_prune_cfg,
       sensitivity='TaylorLossSensitivity',
       search_alg='GreedySearch')
   ```

#### Manual Sparsity

##### Sparsity Process

The API call flow of the channel sparsity feature is shown in the following figure. In this flow, the training environment uses the CPU environment or NPU environment of the PyTorch framework. Based on the inference script of this open-source framework, AMCT APIs are called to complete model compression. The compressed deployment model must be converted into an offline model adapted to the AI processor using the ATC tool before inference can be performed on the AI processor:

![](./en/figures/channel_sparse_interface.png "Channel sparsity API call flow")

The blue parts are implemented by the user, and the gray parts are implemented by calling the APIs provided by AMCT:

1. The user first constructs the original PyTorch model and calls the [create\_prune\_retrain\_model](./en/api/create_prune_retrain_model.md) API to modify the original model, inserting channel sparsity mask operators into the graph structure. The number of parameters of the modified model is pruned.
2. Train the modified model until the accuracy meets the requirement. If the training process is interrupted, based on the original model and the file that records sparsity information, call the [restore\_prune\_retrain\_model](./en/api/restore_prune_retrain_model.md) API again to sparsify the model and continue quantization-aware training until the accuracy meets the requirement.
3. Based on the final retrained channel-sparsity model, generate a **pth** file that meets the accuracy requirement; or call the [save\_prune\_retrain\_model](./en/api/save_prune_retrain_model.md) API to generate the final ONNX simulation model and deployment model.

##### Call Example

> [!NOTE] Note
>
>1. For the steps marked "to be supplemented by the user" in the following example, the user needs to supplement the processing based on their own model and dataset. The example provides only sample code.
>2. For the parts that call AMCT, the function input parameters can be adjusted based on the actual situation. Sparsity is based on the user's training process. Ensure that a training script based on the PyTorch environment is already available and that the accuracy after training is normal.

1. Import the AMCT package and set the log level through the environment variables in [Tool Building](./en/build.md).

   ```python
   import amct_pytorch as amct
   ```

2. <a name="li341817311449"></a>(Optional, to be supplemented by the user) It is recommended to use the original model to be sparsified and the test set to perform inference in the PyTorch environment to verify that the environment and inference script are normal.

   This step is recommended. Ensure that the original model can complete inference with normal accuracy. When performing this step, you can use a subset of the test set to reduce the running time.

   ```python
   ori_model.load()
   # Test the model.
   user_test_model(ori_model, test_data, test_iterations)
   ```

3. Call AMCT to execute the training process with sparsity operators.
   1. Modify the original model to insert channel sparsity mask operators into the graph structure.

      Before implementing this step, restore the trained parameters, such as **ori\_model.load\(\)** in [2](#li341817311449).

      ```python
      simple_cfg = './retrain.cfg'
      record_file = './tmp/record.txt'
      prune_retrain_model = amct.create_prune_retrain_model(model=ori_model,
                                      input_data=ori_model_input_data,
                                      config_defination=simple_cfg,
                                      record_file=record_file)
      ```

   2. (To be supplemented by the user) Use the modified graph to create backward gradients and perform training on the training set.
      1. Use the modified graph to create backward gradients.

         This step must be performed after the sparsity model.

         ```python
         optimizer = user_create_optimizer(prune_retrain_model)
         ```

      2. Restore the model from the trained checkpoint and train the model.

         Note: Restore the model parameters from the trained checkpoint before training.

         ```python
         quant_pth = './ckpt/user_model'
         user_train_model(optimizer, prune_retrain_model, train_data)
         ```

   3. (Optional) If the [save\_prune\_retrain\_model](./en/api/save_prune_retrain_model.md) API is called, refer to this step. If the model is saved as a **pth** file, this step is not required.

      Save the model to implement channel sparsity.

      ```python
      prune_retrain_model = amct.save_prune_retrain_model(
           model=pruned_retrain_model,
           save_path=save_path,
           input_data=input_data)
      ```

4. (To be supplemented by the user) Based on the ONNX Runtime environment, use the channel-sparsified model (**prune\_retrain\_model**) to perform inference on the test set (**test\_data**) and test the accuracy of the quantized simulation model.

   Compare the accuracy of the sparsified simulation model with the original accuracy in [2](#li341817311449) to observe the impact of channel sparsity on accuracy.

   ```python
   prune_retrain_model = './results/user_model_fake_prune_model.onnx'
   user_do_inference_onnx(prune_retrain_model, test_data, test_iterations)
   ```

**If the training process is interrupted and data needs to be restored from the ckpt to continue training, the call flow is as follows:**

1. Prepare the original model.

   ```python
   ori_model= user_create_model()
   ```

2. Call AMCT to restore the quantization process.
   1. Modify the model, insert the channel sparsity mask operator into the graph structure, and save it as a new **prune\_model**.

      ```python
      model = ori_model
      input_data = ori_model_input_data
      record_file = './tmp/record.txt'
      config_defination = './prune_cfg.cfg'
      pth_file = '/your/path/to/save/tmp.pth'
      prune_retrain_model = amct.restore_prune_retrain_model(
          model=model,
          input_data=input_data,
          record_file=record_file,
          config_defination=config_defination,
          pth_file=pth_file,
          state_dict_name='state_dict')

      ```

   2. (To be supplemented by the user) Use the modified model to restore the checkpoint, create the backward gradient, and perform training on the training set.
      1. Restore the model parameters from the checkpoint where the sparsified training was interrupted.

         ```python
         quant_pth = './ckpt/user_prune_model'
         user_train_model(optimizer, prune_retrain_model, train_data)
         ```

      2. Use the modified graph to create backward gradients.

         This step must be performed after restoring the model parameters.

         ```python
         optimizer = user_create_optimizer(prune_retrain_model)
         ```

      3. Restore the model from the trained checkpoint and train the model.

         Note: Restore the model parameters from the trained checkpoint before training.

         ```python
         user_train_model(optimizer, prune_retrain_model, train_data)
         ```

   3. (Optional) If the [save\_prune\_retrain\_model](./en/api/save_prune_retrain_model.md) API is called, refer to this step. This step is not required if the model is saved as a **pth** file.

      Save the model to implement channel sparsity.

      ```python
      prune_retrain_model = amct.save_prune_retrain_model(
           model=pruned_retrain_model,
           save_path=save_path,
           input_data=input_data)
      ```

3. (To be supplemented by the user) Based on the ONNX Runtime environment, use the channel-sparsified model (**prune\_retrain\_model**) to perform inference on the test set (**test\_data**) and test the accuracy of the quantized simulation model.

   Compare the accuracy of the sparsified simulation model with the original accuracy in [2](#li341817311449) to observe the impact of channel sparsity on accuracy.

   ```python
   prune_retrain_model = './results/user_model_fake_prune_model.onnx'
   user_do_inference_onnx(prune_retrain_model, test_data, test_iterations)
   ```

##### Post-Processing

If the model output after sparsity is in the **pth** format, refer to this section. If the [save\_prune\_retrain\_model](./en/api/save_prune_retrain_model.md) API is called, this is not required.

Because the output **pth** model cannot be directly used for inference, you need to convert the **pth** model to an ONNX network model by yourself, or call the [save\_prune\_retrain\_model](./en/api/save_prune_retrain_model.md) API to save it as the final ONNX simulation model and deployment model, and then use the ATC tool for model conversion. The call example of the [save\_prune\_retrain\_model](./en/api/save_prune_retrain_model.md) API is as follows:

```python
prune_retrain_model = amct.save_prune_retrain_model(
     model=pruned_retrain_model,
     save_path=save_path,
     input_data=input_data)
```

##### Manual Tuning

After sparsity, if the accuracy does not meet the requirement, you can tune the model by referring to the methods provided in this section.

If the accuracy still does not meet the requirement after retraining the sparsified model with a certain set of configurations, you can modify the simple configuration file (for file configuration, see [Simple Configuration File for Quantization-Aware Training](./en/context/qat_config.md)) to sparsify and retrain the model again. Common methods include the following:

- Adjust the sparsity rate, which is controlled by the **prune\_ratio** parameter in the simple configuration file. You can try reducing the sparsity rate and sparsifying again for debugging.
- Do not sparsify certain layers, which is controlled by the **regular\_prune\_skip\_layers** parameter in the simple configuration file. Use this parameter to configure the layers that do not need to be sparsified.

### 4:2 Structured Sparsity

#### Sparsity Process

The API call flow of the 4:2 structured sparsity feature is shown in the following figure. In this flow, the training environment uses the CPU or NPU environment of the PyTorch framework. Based on the inference script of this open-source framework, AMCT APIs are called to complete model compression. The compressed deployment model must be converted into an offline model adapted to the AI processor using the ATC tool before inference can be performed on the AI processor:

![](./en/figures/m4n2_interface.png "4:2 structured sparsity API call flow")

The blue parts are implemented by the user, and the gray parts are implemented by calling the APIs provided by AMCT:

1. The user first constructs the original PyTorch model and calls the [create\_prune\_retrain\_model](./en/api/create_prune_retrain_model.md) API to modify the original model, replacing the operators to be sparsified with operators that have 4:2 structured sparsity inserted.
2. Train the modified model until the accuracy meets the requirement. If the training process is interrupted, based on the original model and the file that records sparsity information, call the [restore\_prune\_retrain\_model](./en/api/restore_prune_retrain_model.md) API again to sparsify the model and continue quantization-aware training until the accuracy meets the requirement.
3. Based on the user's final retrained 4:2 structured sparse model, call the [save\_prune\_retrain\_model](./en/api/save_prune_retrain_model.md) API to restore the replaced operators and sparsify the weights, generating the final ONNX simulation model and the deployment model.

#### Call Example

> [!NOTE] Note
>
>1. In the following example, the steps marked "to be supplemented by the user" require the user to supplement the processing based on their own model and dataset. The example provides only sample code.
>2. For the parts that call AMCT, the function input parameters can be adjusted according to the actual situation. Sparsity is based on the user's training process. Ensure that a training script based on the PyTorch environment is available and that the accuracy after training is normal.

1. <a name="li341817311449"></a>(Optional, to be supplemented by the user) It is recommended to use the original model to be quantized and the test set to perform inference in the PyTorch environment to verify that the environment and the inference script are normal.

   It is recommended to perform this step to ensure that the original model can complete inference with normal accuracy. When performing this step, you can use a subset of the test set to reduce the running time.

   ```python
   ori_model.load()
   # Test the model.
   user_test_model(ori_model, test_data, test_iterations)
   ```

2. Call AMCT to execute the 4:2 structured sparsity process.
   1. Modify the original model to replace the operators to be sparsified with operators that have 4:2 structured sparsity inserted.

      Before implementing this step, restore the trained parameters, such as **ori\_model.load\(\)** in the previous step.

      ```python
      simple_cfg = './retrain.cfg'
      record_file = './tmp/record.txt'
      prune_retrain_model = amct.create_prune_retrain_model(model=ori_model,
                                      input_data=ori_model_input_data,
                                      config_defination=simple_cfg,
                                      record_file=record_file)
      ```

   2. (To be supplemented by the user) Use the modified graph to create backward gradients and perform training on the training set.
      1. Use the modified graph to create backward gradients.

         This step must be performed after the sparsified model.

         ```python
         optimizer = user_create_optimizer(prune_retrain_model)
         ```

      2. Restore the model from the trained checkpoint and train the model.

         Note: Restore the model parameters from the trained checkpoint before training.

         ```python
         quant_pth = './ckpt/user_model'
         user_train_model(optimizer, prune_retrain_model, train_data)
         ```

   3. Call the [save\_prune\_retrain\_model](./en/api/save_prune_retrain_model.md) API to save the model, restore the replaced operators, apply structured sparsity to the weights, and generate the final ONNX simulation model and deployment model.

      ```python
      prune_retrain_model = amct.save_prune_retrain_model(
           model=pruned_retrain_model,
           save_path=save_path,
           input_data=input_data)
      ```

3. (Optional, to be supplemented by the user) Based on the ONNX Runtime environment, use the sparsified model (**prune\_retrain\_model**) to perform inference on the test set (**test\_data**) and test the accuracy of the quantized simulation model.

   By comparing the accuracy of the sparsified simulation model with the original accuracy in [1](#li341817311449), you can observe the impact of 4:2 structured sparsity on accuracy.

   ```python
   prune_retrain_model = './results/user_model_fake_prune_model.onnx'
   user_do_inference_onnx(prune_retrain_model, test_data, test_iterations)
   ```

**If the training process is interrupted and data needs to be restored from the checkpoint to continue training, the call flow is as follows:**

1. Prepare the original model.

   ```python
   ori_model= user_create_model()
   ```

2. Call AMCT to restore the quantization-aware training process.
   1. Modify the model by replacing the operators to be sparsified with operators that have 4:2 structured sparsity inserted, and save it as a new **prune\_model**.

      ```python
      model = ori_model
      input_data = ori_model_input_data
      record_file = './tmp/record.txt'
      config_defination = './prune_cfg.cfg'
      pth_file = '/your/path/to/save/tmp.pth'
      prune_retrain_model = amct.restore_prune_retrain_model(
          model=model,
          input_data=input_data,
          record_file=record_file,
          config_defination=config_defination,
          pth_file=pth_file,
          state_dict_name='state_dict')

      ```

   2. (To be supplemented by the user) Use the modified model to restore the checkpoint, create the backward gradient, and perform training on the training set.
      1. Restore the model parameters from the checkpoint where the post-sparsity training was interrupted.

         ```python
         quant_pth = './ckpt/user_prune_model'
         user_train_model(optimizer, prune_retrain_model, train_data)
         ```

      2. Use the modified graph to create the backward gradient.

         This step must be performed after the model parameters are restored.

         ```python
         optimizer = user_create_optimizer(prune_retrain_model)
         ```

      3. Restore the model from the trained checkpoint and train the model.

         Note: Restore the model parameters from the trained checkpoint before training.

         ```python
         user_train_model(optimizer, prune_retrain_model, train_data)
         ```

   3. Call the [save\_prune\_retrain\_model](./en/api/save_prune_retrain_model.md) API to save the model, restore the replaced operators, and apply structured sparsity to the weights, generating the final ONNX simulation model and deployment model.

      ```python
      prune_retrain_model = amct.save_prune_retrain_model(
           model=pruned_retrain_model,
           save_path=save_path,
           input_data=input_data)
      ```

3. (Optional, to be supplemented by the user) Based on the ONNX Runtime environment, use the sparsified model (**prune\_retrain\_model**) to perform inference on the test set (**test\_data**) and test the accuracy of the quantized simulation model.

   By comparing the accuracy of the sparsified simulation model with the original accuracy in [1](#li341817311449), you can observe the impact of 4:2 structured sparsity on accuracy.

   ```python
   prune_retrain_model = './results/user_model_fake_prune_model.onnx'
   user_do_inference_onnx(prune_retrain_model, test_data, test_iterations)
   ```

#### Manual Tuning

After sparsity, if the accuracy does not meet the requirement, you can tune the model by referring to the methods provided in this section.

If the accuracy of the sparsified model still does not meet the requirement after retraining with a certain set of configurations, you can modify the simple configuration file (for file configuration, see [Quantization-Aware Training Simplified Configuration File](./en/context/qat_config.md)) to sparsify the model again and retrain it. Common methods include the following:

- Adjust the update interval (the interval for calculating which elements are retained), which is controlled by the **update\_freq** parameter in the simple configuration file. You can try changing the update interval to a positive integer and sparsify the model again for debugging. Generally, a smaller update interval yields greater accuracy gains.
- Exclude certain layers from sparsity, which is controlled by the **regular\_prune\_skip\_layers** parameter in the simple configuration file. Use this parameter to configure the layers that do not need to be sparsified.

### Combined Compression

Combined compression applies both quantization and sparsity to the same network model. This section provides a detailed introduction to the specific combined compression methods supported.

#### Combined Compression Method

Currently, combined compression supports the following combination methods. When using AMCT for compression, each compressible operator in each layer can select only one combined compression method at a time. The brief flow is shown in the following figure.

- [Manual Sparsity](#manual-sparsity)+[Quantization-Aware Training](#quantization-aware-training)  INT8 quantization
- [4:2 Structured Sparsity](#42-structured-sparsity)+[Quantization-Aware Training](#quantization-aware-training)  INT8 quantization

Currently, the compression configuration of the combined compression feature is handled manually by the user (therefore it is also called static combined compression; for the configuration method of the compression configuration file, see the [simple configuration file for quantization-aware training](./en/context/qat_config.md)). By manually setting the global quantization bit width and sparsity rate (channel sparsity rate) or updating the interval of 4:2 sparsity, automatic model compression is achieved.

For the layers that support combined compression and the constraints, see [Manual Sparsity](#manual-sparsity), [4:2 Structured Sparsity](#42-structured-sparsity), and [Quantization-Aware Training](#quantization-aware-training), respectively.

![](./en/figures/combined_compress.png "Brief flow of combined compression")

#### Combined Compression Scenarios

The current combined compression supports the following scenarios. In actual use, they can be controlled through parameters in the simple configuration file. In the following scenarios, whole-network quantization refers to [Quantization-Aware Training](#quantization-aware-training), where:

- Whole-network quantization: choose one of the two quantization methods.

  - Whole-network (global) quantization configuration parameter: retrain\_data\_quant\_config/retrain\_weight\_quant\_config
  - Differentiated configuration parameter for some layers: **override\_layer\_configs** or **override\_layer\_types**

    Parameter priority: **override\_layer\_configs** \> **override\_layer\_types** \> **retrain\_data\_quant\_config/retrain\_weight\_quant\_config**

- Whole-network sparsity: includes channel sparsity and 4:2 structured sparsity. Choose one of the two when using.

  - Whole-network (global) sparsity configuration parameter: **prune\_config**
  - Differentiated sparsity parameter for some layers: **override\_layer\_configs** or **override\_layer\_types**

    Parameter priority: **override\_layer\_configs** \> **override\_layer\_types** \> **retrain\_data\_quant\_config/retrain\_weight\_quant\_config**

For detailed explanation of the above parameters, see [Simple Configuration File for Quantization-Aware Training](./en/context/qat_config.md).

**Table 1**  Combined compression scenarios

<a name="zh-cn_topic_0000001265077702_table1841654916270"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002517028698_row3417134942711"><th class="cellrowborder" valign="top" width="19.14191419141914%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000002517028698_p641794972716"><a name="zh-cn_topic_0000002517028698_p641794972716"></a><a name="zh-cn_topic_0000002517028698_p641794972716"></a>Combined Compression Scenario</p></th>
<th class="cellrowborder" valign="top" width="51.27512751275128%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000002517028698_p17417114911276"><a name="zh-cn_topic_0000002517028698_p17417114911276"></a><a name="zh-cn_topic_0000002517028698_p17417114911276"></a>Configuration Parameters</p></th>
<th class="cellrowborder" valign="top" width="29.58295829582958%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000002517028698_p146370141022"><a name="zh-cn_topic_0000002517028698_p146370141022"></a><a name="zh-cn_topic_0000002517028698_p146370141022"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002517028698_row19417549112711"><td class="cellrowborder" valign="top" width="19.14191419141914%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002517028698_p124171149182711"><a name="zh-cn_topic_0000002517028698_p124171149182711"></a><a name="zh-cn_topic_0000002517028698_p124171149182711"></a>Whole-network quantization + whole-network sparsity</p></td>
<td class="cellrowborder" valign="top" width="51.27512751275128%" headers="mcps1.2.4.1.2 "><a name="zh-cn_topic_0000002517028698_ul189333368587"></a><a name="zh-cn_topic_0000002517028698_ul189333368587"></a><ul id="zh-cn_topic_0000002517028698_ul189333368587"><li>Quantization parameters: retrain_data_quant_config/retrain_weight_quant_config</li><li>Sparsity parameters: prune_config</li></ul>
<p id="zh-cn_topic_0000002517028698_p11121392577"><a name="zh-cn_topic_0000002517028698_p11121392577"></a><a name="zh-cn_topic_0000002517028698_p11121392577"></a>Do not configure override_layer_configs or override_layer_types</p></td>
<td class="cellrowborder" valign="top" width="29.58295829582958%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002517028698_p10387131415315"><a name="zh-cn_topic_0000002517028698_p10387131415315"></a><a name="zh-cn_topic_0000002517028698_p10387131415315"></a>-</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028698_row6418104912278"><td class="cellrowborder" valign="top" width="19.14191419141914%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002517028698_p2418104912271"><a name="zh-cn_topic_0000002517028698_p2418104912271"></a><a name="zh-cn_topic_0000002517028698_p2418104912271"></a>Partial-layer differentiated quantization + whole-network sparsity</p></td>
<td class="cellrowborder" valign="top" width="51.27512751275128%" headers="mcps1.2.4.1.2 "><a name="zh-cn_topic_0000002517028698_ul1961884514588"></a><a name="zh-cn_topic_0000002517028698_ul1961884514588"></a><ul id="zh-cn_topic_0000002517028698_ul1961884514588"><li>Quantization parameters: retrain_data_quant_config/retrain_weight_quant_config + override_layer_configs or override_layer_types</li><li>Sparsity parameters: prune_config</li></ul></td>
<td class="cellrowborder" valign="top" width="29.58295829582958%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002517028698_p18418104922711"><a name="zh-cn_topic_0000002517028698_p18418104922711"></a><a name="zh-cn_topic_0000002517028698_p18418104922711"></a>When configuring feature parameters, specify the global configuration first; otherwise, the feature is not enabled.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028698_row104181492272"><td class="cellrowborder" valign="top" width="19.14191419141914%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002517028698_p1041811493272"><a name="zh-cn_topic_0000002517028698_p1041811493272"></a><a name="zh-cn_topic_0000002517028698_p1041811493272"></a>Whole-network quantization + partial-layer differentiated sparsity</p></td>
<td class="cellrowborder" valign="top" width="51.27512751275128%" headers="mcps1.2.4.1.2 "><a name="zh-cn_topic_0000002517028698_ul7935131418596"></a><a name="zh-cn_topic_0000002517028698_ul7935131418596"></a><ul id="zh-cn_topic_0000002517028698_ul7935131418596"><li>Quantization parameters: retrain_data_quant_config/retrain_weight_quant_config</li><li>Sparsity parameters: prune_config + override_layer_configs or override_layer_types</li></ul></td>
<td class="cellrowborder" valign="top" width="29.58295829582958%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002517028698_p113296231735"><a name="zh-cn_topic_0000002517028698_p113296231735"></a><a name="zh-cn_topic_0000002517028698_p113296231735"></a>When configuring feature parameters, specify the global configuration first; otherwise, the feature is not enabled.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028698_row2418104914273"><td class="cellrowborder" valign="top" width="19.14191419141914%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002517028698_p441844932719"><a name="zh-cn_topic_0000002517028698_p441844932719"></a><a name="zh-cn_topic_0000002517028698_p441844932719"></a>Partial-layer quantization + partial-layer differentiated sparsity</p></td>
<td class="cellrowborder" valign="top" width="51.27512751275128%" headers="mcps1.2.4.1.2 "><a name="zh-cn_topic_0000002517028698_ul19449126706"></a><a name="zh-cn_topic_0000002517028698_ul19449126706"></a><ul id="zh-cn_topic_0000002517028698_ul19449126706"><li>Quantization parameters: retrain_data_quant_config/retrain_weight_quant_config + override_layer_configs or override_layer_types</li><li>Sparsity parameters: prune_config + override_layer_configs or override_layer_types</li></ul></td>
<td class="cellrowborder" valign="top" width="29.58295829582958%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002517028698_p4158172415318"><a name="zh-cn_topic_0000002517028698_p4158172415318"></a><a name="zh-cn_topic_0000002517028698_p4158172415318"></a>When configuring feature parameters, specify the global configuration first; otherwise, the feature is not enabled.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028698_row141864913279"><td class="cellrowborder" valign="top" width="19.14191419141914%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002517028698_p17418154942719"><a name="zh-cn_topic_0000002517028698_p17418154942719"></a><a name="zh-cn_topic_0000002517028698_p17418154942719"></a>Whole-network quantization</p></td>
<td class="cellrowborder" valign="top" width="51.27512751275128%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002517028698_p1969818452012"><a name="zh-cn_topic_0000002517028698_p1969818452012"></a><a name="zh-cn_topic_0000002517028698_p1969818452012"></a>retrain_data_quant_config/retrain_weight_quant_config</p>
<p id="zh-cn_topic_0000002517028698_p151467471206"><a name="zh-cn_topic_0000002517028698_p151467471206"></a><a name="zh-cn_topic_0000002517028698_p151467471206"></a>Do not configure override_layer_configs or override_layer_types.</p></td>
<td class="cellrowborder" valign="top" width="29.58295829582958%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002517028698_p041864916271"><a name="zh-cn_topic_0000002517028698_p041864916271"></a><a name="zh-cn_topic_0000002517028698_p041864916271"></a>-</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028698_row0418449192712"><td class="cellrowborder" valign="top" width="19.14191419141914%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002517028698_p5418249162714"><a name="zh-cn_topic_0000002517028698_p5418249162714"></a><a name="zh-cn_topic_0000002517028698_p5418249162714"></a>Differentiated quantization for some layers</p></td>
<td class="cellrowborder" valign="top" width="51.27512751275128%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002517028698_p341919495279"><a name="zh-cn_topic_0000002517028698_p341919495279"></a><a name="zh-cn_topic_0000002517028698_p341919495279"></a>retrain_data_quant_config/retrain_weight_quant_config+override_layer_configs or override_layer_types</p></td>
<td class="cellrowborder" valign="top" width="29.58295829582958%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002517028698_p135712271031"><a name="zh-cn_topic_0000002517028698_p135712271031"></a><a name="zh-cn_topic_0000002517028698_p135712271031"></a>When configuring feature parameters, specify the global configuration first; otherwise, the feature is not enabled.</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028698_row74191249152720"><td class="cellrowborder" valign="top" width="19.14191419141914%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002517028698_p104194493272"><a name="zh-cn_topic_0000002517028698_p104194493272"></a><a name="zh-cn_topic_0000002517028698_p104194493272"></a>Whole-network sparsity</p></td>
<td class="cellrowborder" valign="top" width="51.27512751275128%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002517028698_p3419849142717"><a name="zh-cn_topic_0000002517028698_p3419849142717"></a><a name="zh-cn_topic_0000002517028698_p3419849142717"></a>prune_config</p>
<p id="zh-cn_topic_0000002517028698_p250381119119"><a name="zh-cn_topic_0000002517028698_p250381119119"></a><a name="zh-cn_topic_0000002517028698_p250381119119"></a>Do not configure override_layer_configs or override_layer_types</p></td>
<td class="cellrowborder" valign="top" width="29.58295829582958%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002517028698_p124190499278"><a name="zh-cn_topic_0000002517028698_p124190499278"></a><a name="zh-cn_topic_0000002517028698_p124190499278"></a>-</p></td>
</tr>
<tr id="zh-cn_topic_0000002517028698_row10181133810426"><td class="cellrowborder" valign="top" width="19.14191419141914%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002517028698_p181811138154214"><a name="zh-cn_topic_0000002517028698_p181811138154214"></a><a name="zh-cn_topic_0000002517028698_p181811138154214"></a>Differentiated sparsity for some layers</p></td>
<td class="cellrowborder" valign="top" width="51.27512751275128%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002517028698_p137714231911"><a name="zh-cn_topic_0000002517028698_p137714231911"></a><a name="zh-cn_topic_0000002517028698_p137714231911"></a>prune_config+override_layer_configs or override_layer_types</p></td>
<td class="cellrowborder" valign="top" width="29.58295829582958%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002517028698_p187228733"><a name="zh-cn_topic_0000002517028698_p187228733"></a><a name="zh-cn_topic_0000002517028698_p187228733"></a>When configuring feature parameters, specify the global configuration first; otherwise, the feature is not enabled.</p></td>
</tr>
</tbody>
</table>

#### Compression Process

The API call flow of combined compression is shown in the following figure. In this flow, the training environment uses the CPU or NPU environment of the PyTorch framework. Based on the inference script of this open-source framework, AMCT APIs are called to complete model compression. The compressed deployment model must be converted into an offline model adapted to the AI processor using the ATC tool before inference can be performed on the AI processor:

![](./en/figures/combined_compress_interface.png "Combined compression API call flow")

The blue part is implemented by the user, and the gray part is implemented by calling the APIs provided by AMCT:

1. The user constructs the original PyTorch model and calls the [create\_compressed\_retrain\_model](./en/api/create_compressed_retrain_model.md) API to modify the original model. The modified model contains operators related to sparsity and quantization.
2. Train the modified model. If the training is not interrupted, perform inference on the trained model. During inference, the quantization factors are written to the record file, and then the [save\_compressed\_retrain\_model](./en/api/save_compressed_retrain_model.md) API is called to save the accuracy simulation model and the deployment model. If the training is interrupted, based on the saved **pth** model parameters, call the [restore\_compressed\_retrain\_model](./en/api/restore_compressed_retrain_model.md) API again to output the model that has been sparsified according to the sparsity relationship and has quantization operators inserted, reload the weights of the model saved before the interruption, continue training, then perform inference, and finally call the [save\_compressed\_retrain\_model](./en/api/save_compressed_retrain_model.md) API to save the quantized model.

#### Call Example

> [!NOTE] Note
>
>1. For training in the PyTorch environment, currently only multi-device training in distributed mode (that is, DistributedDataParallel mode) is supported. Multi-device training in DataParallel mode is not supported, and training in DP mode will report an error.
>2. For the part that calls AMCT, the function input parameters can be adjusted according to the actual situation. Combined compression is based on the user's training process. Ensure that you already have a script for training in the PyTorch environment and that the accuracy after training is normal.
>3. When using the quantization-aware training feature of AMCT, if the training process hangs, check whether other ONNX Runtime programs are running on the current server (you can use the **top** command to view all processes on the server). If so, pause the other ONNX Runtime programs first and then re-execute quantization-aware training.
>4. For the steps marked "to be supplemented by the user" in the following example, the user needs to supplement the processing according to their own model and dataset. The example contains only sample code.

1. Import the AMCT package and set the log level through the environment variables in [Tool Building](./en/build.md).

    ```python
    import amct_pytorch as amct
    ```

2. <a name="li2796152115712"></a>(Optional, to be supplemented by the user) It is recommended to use the original model to be quantized and the test set to perform inference in the PyTorch environment to verify that the environment and the inference script are normal.

    It is recommended to perform this step to ensure that the original model can complete inference with normal accuracy. When performing this step, you can use a subset of the test set to reduce the running time.

    ```python
    ori_model.load()
    # Test the model.
    user_test_model(ori_model, test_data, test_iterations)
    ```

3. Call AMCT to execute the combined compression process.
    1. <a name="li7715155181310"></a>Modify the model, apply sparsity to the model **ori\_model** and insert quantization-related operators, and save it as a new training model (**retrain\_model**).

        Before implementing this step, restore the trained parameters, such as **ori\_model.load\(\)** in [2](#li2796152115712).

        ```python
        simple_cfg = './compressed.cfg'
        record_file = './tmp/record.txt'
        compressed_retrain_model = amct.create_compressed_retrain_model(
                                        model=ori_model,
                                        input_data=ori_model_input_data,
                                        config_defination=simple_cfg,
                                        record_file=record_file)
        ```

    2. (To be supplemented by the user) Use the modified graph to create backward gradients, train on the training set, and train the quantization factors.
        1. Use the modified graph to create backward gradients.

            This step must be performed after [3.a](#li7715155181310).

            ```python
            optimizer = user_create_optimizer(compressed_retrain_model)
            ```

        2. Restore the model from the trained checkpoint and train the model.

            Note: Restore the model parameters from the trained checkpoint before training. The parameters saved during training must include the quantization factors: the quantization factors are generated after the first **batch\_num** training iterations. If the number of training iterations is fewer than **batch\_num**, the process fails.

            ```python
            compressed_pth = './ckpt/user_model'
            user_train_model(optimizer, compressed_retrain_model, train_data)
            ```

        3. After training is complete, perform inference to compute and save the quantization factors.

            ```python
            user_infer_graph(compressed_retrain_model)
            ```

    3. Save the model.

        ```python
        save_path = '/.result/user_model'
        amct.save_compressed_retrain_model(
             model=compressed_retrain_model,
             record_file=record_file,
             save_path=save_path,
             input_data=ori_model_input_data)
        ```

4. (Optional, to be supplemented by the user) In an ONNX Runtime environment, perform inference on the test set (**test\_data**) using the combined-compressed model (**compressed\_model**) to test the accuracy of the combined-compressed simulation model. Compare the accuracy of the combined-compressed simulation model with the original accuracy in [1](#li2796152115712) to observe the impact of combined compression on accuracy.

    ```python
    compressed_model = './results/user_model_fake_quant_model.onnx'
    user_do_inference_onnx(compressed_model, test_data, test_iterations)
    ```

**If the training process is interrupted and data needs to be restored from the ckpt to continue training, the call flow is as follows:**

1. Import the AMCT package and set the log level through the environment variables in [Tool Building](./en/build.md).

    ```python
    import amct_pytorch as amct
    ```

2. Prepare the original model.

    ```python
    ori_model = user_create_model()
    ```

3. Call AMCT to execute the combined compression process.
    1. Modify the model, apply sparsity to the model **ori\_model** and insert quantization-related operators, load the weight parameters, and save it as a new training model (**retrain\_model**).

        ```python
        simple_cfg = './compressed.cfg'
        record_file = './tmp/record.txt'
        compressed_pth_file = './ckpt/user_model_newest.ckpt'
        compressed_retrain_model = amct.restore_compressed_retrain_model(
                                        model=ori_model,
                                        input_data=ori_model_input_data,
                                        config_defination=simple_cfg,
                                        record_file=record_file,
                                        pth_file=compressed_pth_file)
        ```

    2. (To be supplemented by the user) Use the modified graph to create backward gradients, perform training on the training set, and train the quantization factors.
        1. Use the modified graph to create backward gradients.

            This step must be performed after [3.a](#li7715155181310).

            ```python
            optimizer = user_create_optimizer(compressed_retrain_model)
            ```

        2. Restore the model from the trained checkpoint and train the model.

            Note: Restore the model parameters from the trained checkpoint before training. The parameters saved during training must include the quantization factors: quantization factors are generated after the first **batch\_num** training iterations. If the number of training iterations is fewer than **batch\_num**, the process fails.

            ```python
            compressed_pth = './ckpt/user_model'
            user_train_model(optimizer, compressed_retrain_model, train_data)
            ```

        3. After training is complete, perform inference to compute and save the quantization factors.

            ```python
            user_infer_graph(compressed_retrain_model)
            ```

    3. Save the model.

        ```python
        save_path = '/.result/user_model'
        amct.save_compressed_retrain_model(
             model=compressed_retrain_model,
             record_file=record_file,
             save_path=save_path,
             input_data=ori_model_input_data)
        ```

4. (Optional, to be supplemented by the user) In an ONNX Runtime environment, use the combined-compressed model (**compressed\_model**) to perform inference on the test set (**test\_data**) and test the accuracy of the combined-compressed simulation model. Compare the accuracy of the combined-compressed simulation model with the original accuracy in [1](#li2796152115712) to observe the impact of combined compression on accuracy.

    ```python
    compressed_model = './results/user_model_fake_quant_model.onnx'
    user_do_inference_onnx(compressed_model, test_data, test_iterations)
    ```

#### Manual Tuning

If the model accuracy does not meet the requirements after combined compression, perform tuning by referring to the manual tuning section under [Sparsity](#sparsity) and the section under [Quantization-Aware Training](#quantization-aware-training), respectively.

## Pruning

AMCT provides the structured pruning feature `amct_pytorch.pruning` in eager mode: without first converting to an ONNX graph, it directly prunes the dense FFN intermediate dimensions, CNN channels, and MoE experts of an instantiated `torch.nn.Module` in place, while rewriting the related dimensions in the model config. The pruned model can be directly saved and loaded, and can also be further quantized. For detailed configuration and usage, see [Structured Pruning Usage Guide](../amct_pytorch/pruning/README_en.md).

### Pruning Process

The structured pruning API call flow is shown in the following figure. The entire process is completed in the CPU or NPU environment of PyTorch, without any model conversion steps:

![](./en/figures/pruning_interface.png "Structured pruning API call flow")

The blue parts are implemented by the user, and the gray parts are implemented by the user calling the APIs provided by AMCT:

1. The user prepares the original model with loaded weights and a small amount of calibration data (a few forward passes are sufficient). Optional: first call the [prune\_diagnose](./en/api/prune_diagnose.md) API to perform a dry-run to confirm the prunable targets in the model and the estimated benefits. This API does not modify the original model.
2. Call the [prune](./en/api/prune.md) API to perform pruning. When no search parameters are specified, pruning is performed once at the fixed pruning rate in the configuration without copying the model. When `tolerance`, `size_budget`, or a menu configuration is specified, search is entered, and each candidate pruning rate is trial-pruned and evaluated on a full model copy, and the pruning rate that meets the requirements is finally applied. For ready-to-use preset configurations, see the "Predefined Configurations" section in [prune](./en/api/prune.md).
3. Pruning takes effect in place on the original model, and the related dimensions in the **config** are rewritten at the same time. What is pruned and the change in the number of parameters are recorded in `PruneReport`. If accuracy needs to be restored, call the [prune\_finetune](./en/api/prune_finetune.md) API for lightweight fine-tuning. If further compression is needed, the quantization process can be continued after pruning.

### Call Example

> [!NOTE] Note
>
> - In the following examples, the steps marked "to be supplemented by the user" need to be supplemented by the user according to their own model and dataset. The examples provide only sample code.
> - For the parts of the following examples that call AMCT, adjust the function input parameters according to the actual situation.

1. Import the AMCT package, and set the log level through the environment variables in [Tool Building](./en/build.md).

   ```python
   import amct_pytorch as amct
   from amct_pytorch.pruning import PruneReport
   ```

2. (Optional, to be supplemented by the user) Verify the inference script and environment in the original PyTorch environment.

   It is recommended to use the original model to be pruned and the test set to perform inference in the PyTorch environment to verify that the environment and inference script work properly.

   This step is recommended. Ensure that the original model can complete inference with normal accuracy. When performing this step, you can use a subset of the test set to reduce the running time.

   ```python
   user_do_inference_torch(ori_model, test_data, test_iterations)
   ```

3. (To be supplemented by the user) Prepare the calibration data.

   Pruning selects the structures to be removed based on activation statistics, so a small amount of forward input that can represent the deployment scenario is required, usually a few batches.

   ```python
   calib_data = user_prepare_calib_data(batch_num=8)
   ```

4. Call AMCT to prune the model.

   Call the [prune](./en/api/prune.md) API to prune the instantiated model in place. Pass `report` to obtain what was pruned this time and the parameter count changes. The preset configuration can be used directly. For details, see the "Predefined Configurations" section in [prune](./en/api/prune.md).

   ```python
   cfg = {"methods": {"dense": {"name": "low_variance",
                                "kwargs": {"prune_ratio": 0.5}}}}
   report = PruneReport()
   amct.prune(ori_model, cfg, data=calib_data, report=report)
   print(report.as_dict())
   ```

5. (Optional, to be supplemented by the user) Verify the accuracy of the pruned model and save the model.

   Pruning has already rewritten the relevant dimensions in the model config, so the model can be saved directly without additional model conversion steps.

   ```python
   user_do_inference_torch(ori_model, test_data, test_iterations)
   ori_model.save_pretrained('./pruned_model')
   ```

## Layer-by-Layer Distillation

AMCT provides a graph-based quantization distillation method for torch models. This method converts a torch model into an ONNX model to obtain the graph structure of the torch model, performs distillation optimization based on the parsed graph structure, and finally generates an ONNX quantized model.

This section describes in detail the layers supported by layer-by-layer distillation, the API call flow, and examples.

- Operators that support distillation and quantization:
  - torch.nn.Linear: Reused layers (sharing weight and bias parameters) do not support quantization.
  - torch.nn.Conv2d: Quantization is supported only when **padding\_mode** is **zeros**. Reused layers (sharing the **weight** and **bias** parameters) do not support quantization.

- Activation operators that support distillation:
  - torch.nn.ReLU
  - torch.nn.LeakyReLU
  - torch.nn.Sigmoid
  - torch.nn.Tanh
  - torch.nn.Softmax

- Normalization operators that support distillation

  **torch.nn.BatchNorm2d**: Reused layers (sharing the **weight** and **bias** parameters) do not support distillation.

### Distillation Process

The call flow is shown in the following figure.

![](./en/figures/distill_interface.png "Distillation API call flow")

The blue part is implemented by the user, and the gray part is implemented by the user calling the APIs provided by AMCT. The layer-by-layer distillation feature consists of four parts: creating the distillation configuration, creating the distilled model, distilling the quantized model, and saving the quantized model. The details are as follows:

1. Create the distillation configuration. The user first constructs the original PyTorch model, and then uses the [create\_distill\_config](./en/api/create_distill_config.md) API to combine the user-defined distillation configuration with the distillation configuration defined by the AMCT algorithm, outputting the distillation configuration information of each network layer.
2. Create the distilled model. Call the [create\_distill\_model](./en/api/create_distill_model.md) API to modify the original model and generate a quantized model to be distilled based on the distillation configuration information.
3. Distill the model. Call the [distill](./en/api/distill.md) API to perform block-by-block distillation on the network based on the model inference and optimization methods configured by the user, combined with the distillation configuration information, to obtain a performance-optimized quantized model after distillation.
4. Save the quantized model. Finally, call the [save\_distill\_model](./en/api/save_distill_model.md) API to save the distilled and optimized quantized model, including the model file that can be used for accuracy simulation in the ONNX Runtime execution framework and the model file that can be deployed on the AI processor.

### Call Example

> [!NOTE] Note
>
>- In the following examples, the steps marked with "to be supplemented by the user" need to be supplemented by the user based on their own model and dataset. The examples only provide sample code.
>- For the parts that call AMCT in the following examples, adjust the function input parameters according to the actual situation.

1. Import the AMCT package and set the log level through the environment variables in [Tool Building](./en/build.md).

   ```python
   import amct_pytorch as amct
   ```

2. <a name="li13259141318410"></a>(Optional, to be supplemented by the user) Verify the inference script and environment in the original PyTorch environment.

   It is recommended to use the original model to be quantized and the test set to perform inference in the PyTorch environment to verify that the environment and inference script work properly.

   It is recommended to perform this step to ensure that the original model can complete inference with normal accuracy. When performing this step, you can use a subset of the test set to reduce the runtime.

   ```python
   user_do_inference_torch(ori_model, test_data, test_iterations)
   ```

3. Call AMCT to distill the model.
   1. Call the [create\_distill\_config](./en/api/create_distill_config.md) API to generate the distillation configuration. The output distillation configuration contains both user-defined and automatically discovered distillation structures.

      ```python
      config_file = './tmp/config.json'
      simple_cfg = './distill.cfg'
      amct.create_distill_config(config_file=config_file,
                                   model=ori_model,
                                   input_data=ori_model_input_data,
                                   config_defination=simple_cfg)
      ```

   2. Call the [create\_distill\_model](./en/api/create_distill_model.md) API to create the distilled model.

      Quantize the floating-point model to be distilled, and replace the operators to be compressed in the floating-point model with CANN quantization operators.

      ```python
      compress_model = amct.create_distill_model(
                       config_file=config_file,
                       model=ori_model,
                       input_data=ori_model_input_data)
      ```

   3. Perform layer-by-layer distillation.

      Call the [distill](./en/api/distill.md) API to perform layer-by-layer distillation. Distillation is performed on the distillation structures in the configuration.

      ```python
      distill_model = amct.distill(
                      model=ori_model,
                      compress_model,
                      config_file=config_file,
                      train_loader,
                      epochs=1,
                      lr=1e-3,
                      sample_instance=None,
                      loss=loss,
                      optimizer=optimizer)
      ```

   4. Save the distilled model.

      Call the [save\_distill\_model](./en/api/save_distill_model.md) API to insert operators such as AscendQuant and AscendDequant, and save the result as a distilled model.

      ```python
      amct.save_distill_model(
                       model,
                       "./model/distilled",
                       input_data,
                       record_file="./results/records.txt",
                       input_names=['input'],
                       output_names=['output'],
                       dynamic_axes={'input':{0: 'batch_size'},
                                     'output':{0: 'batch_size'}})
      ```

4. (Optional, to be supplemented by the user) Based on the ONNX Runtime environment, run inference on the test set (**test\_data**) using the distilled model (**quant\_model**) to test the accuracy of the distilled simulation model.

   By comparing the accuracy of the quantized simulation model with the original accuracy in [2](#li13259141318410), you can observe the impact of quantization on accuracy.

   ```python
   quant_model = './results/user_model_fake_quant_model.onnx'
   user_do_inference_onnx(quant_model, test_data, test_iterations)
   ```

## Tensor Decomposition

Tensor decomposition reduces inference overhead by decomposing the convolution kernel tensor and converting one convolution into a stack of two smaller convolutions. It is recommended when a user model contains a large number of convolutions and the convolution kernel shapes are generally larger than \(64, 64, 3, 3\).

Currently, only convolutions that meet the following conditions can be decomposed:

- group=1, dilation=\(1,1\), stride<3
- kernel\_h\>2, kernel\_w\>2

Whether a convolution is decomposed is automatically determined by AMCT. Even if the preceding conditions are met, decomposition is not guaranteed. For example, if the original PyTorch model used by the user contains a `torch.nn.Conv2d` layer that meets the preceding conditions, the corresponding `Conv2d` layer may be decomposed into two `Conv2d` layers.

Generally, the accuracy of the decomposed model is lower than that of the original model. Therefore, after decomposition, fine-tune is required to improve accuracy. Finally, use AMCT to convert the decomposed model into a quantized model that can be deployed on AI processors, so as to achieve better performance during model inference.

This scenario is optional, and users can decide whether to decompose the original model.

### Decomposition Constraints

If the weight shape of a torch.nn.Conv2d layer is too large, the decomposition may take too long or terminate abnormally. To prevent this, before performing the decomposition, refer to the following constraints or reference data:

- **Decomposition tool performance reference data:**

  - CPU: Intel\(R\) Xeon\(R\) CPU E5-2699 v4 @ 2.20GHz
  - Memory: 512G

    Decomposing a single convolution layer:

  - shape\(512, 512, 5, 5\), taking about 7 seconds.
  - shape\(1024, 1024, 3, 3\), taking about 12 seconds.
  - shape\(1024, 1024, 5, 5\), taking about 52 seconds.
  - shape\(2048, 2048, 3, 3\), taking about 89 seconds.
  - shape\(2048, 2048, 5, 5\), approximately 374 seconds.

- **Memory overflow risk reminder**:

    Reference values for memory overflow risk when decomposing large convolution kernels: decomposing a convolution kernel with shape \(2048, 2048, 5, 5\) requires approximately 9 GB of memory.

### Decomposition Methods

Tensor decomposition supports two usage modes: online tensor decomposition and offline tensor decomposition. You can select either mode for decomposition based on your actual situation. The details are as follows:

1. **Online tensor decomposition**

    This mode directly decomposes the original network model and then performs fine-tuning: introduce the tensor decomposition API into the training code to decompose the model that already contains pretrained weights, and then perform fine-tuning.

    In this process, the model structure is directly modified and the weights are updated during tensor decomposition, so the decomposed model can be used directly. Its advantage is ease of use, requiring only one step; its disadvantage is that the decomposition computation must be performed again each time the script is invoked, which takes a certain amount of time.

2. **Offline tensor decomposition**

    This mode first decomposes the original network model and then performs fine-tuning: first, call the tensor decomposition API on the model containing pretrained weights to decompose it, and save the decomposition information file and the decomposed weights. For subsequent use, introduce the tensor decomposition API into the training script, read the saved decomposition information file to decompose the model structure, then load the saved decomposed weights, and finally perform fine-tuning.

    In this process, the decomposition information and the decomposed weights are saved first and then quickly loaded during fine-tuning. Its advantage is that decomposition can be performed once and reused multiple times, with loading taking almost no time; its disadvantage is that it requires two steps, and the decomposed weights must be saved and loaded during use.

### Decomposition Process

The API call flow is shown in the following figure.

![](./en/figures/decomposition_interface.png "Tensor decomposition API call flow")

1. **Online decomposition process**

    In the training script, prepare a **torch.nn.Module** model object containing pretrained weights. Before passing the model parameters to the optimizer, pass the model object to the [auto\_decomposition](./en/api/auto_decomposition.md) API for tensor decomposition to obtain the decomposed model object, which can then be directly fine-tuned.

2. **Offline decomposition process**

    1. <a name="li265516506413"></a>In any script, prepare a **torch.nn.Module** model object containing pretrained weights. Pass the model object and the save path of the decomposition information file to the [auto\_decomposition](./en/api/auto_decomposition.md) API for tensor decomposition to obtain the decomposed model object and the saved decomposition information file, and then save the decomposed model weights.
    2. <a name="li1228813421049"></a>During fine-tuning, in the training script, before passing the model parameters to the optimizer, pass the model object and the path of the decomposition information file obtained in [2.a](#li265516506413) to the [decompose\_network](./en/api/decompose_network.md) API. This API modifies the model structure to the decomposed structure, then loads the decomposed model weights saved in [2.a](#li265516506413), and fine-tunes the model.

    > [!NOTE] Note
    >During offline decomposition, after calling the [auto\_decomposition](./en/api/auto_decomposition.md) API in step [2.a](#li265516506413), you need to save the decomposed model weights by yourself. After calling [decompose\_network](./en/api/decompose_network.md) in step [2.b](#li1228813421049), you need to load the saved decomposed model weights by yourself.
    >This design allows you to freely control the storage and retrieval of weight files, for example, by storing custom information in the weight file.

After tensor decomposition, one convolution is decomposed into two cascaded convolutions. The following figure shows one convolution layer in the model before and after decomposition.

![](./en/figures/conv_decompose.png "Convolution before and after decomposition")

### Call Example

> [!NOTE] Note
> In the following examples, \(\*\) indicates existing user code, and ... indicates omitted existing user code. These are examples only; actual user code may differ. Adjust them based on the actual situation.

1. **Online tensor decomposition**

    In the training script, call [auto\_decomposition](./en/api/auto_decomposition.md) to decompose a PyTorch model with pretrained weights, and then fine-tune it directly.

    ```python
    from amct_pytorch.tensor_decompose import auto_decomposition
    net = Net()                                          # (*) Build the model object.
    net.load_state_dict(torch.load("src_path/net.pth"))  # (*) Load the model weights.
    net, changes = auto_decomposition(model=net)         # Perform tensor decomposition.
    optimizer = build_optimizer(net, ...)                # (*) Build the optimizer (pass the model parameters to the optimizer).
    train(net, optimizer, ...)                           # (*) fine-tune
    ```

2. **Offline tensor decomposition**
    1. In any script, call [auto\_decomposition](./en/api/auto_decomposition.md) to decompose a PyTorch model with pretrained weights, and save the decomposition information file and the decomposed model weights.

        ```python
        from amct_pytorch.tensor_decompose import auto_decomposition
        net = Net()                                                                 # (*) Build the model object.
        net.load_state_dict(torch.load("src_path/weights.pth"))                     # (*) Load the model weights.
        net, changes = auto_decomposition(                                          # Perform tensor decomposition and save the decomposition information file.
            model=net,
            decompose_info_path="decomposed_path/decompose_info.json"               # Path for saving the decomposition information file.
        )
        torch.save(net.state_dict(), "decomposed_path/decomposed_weights.pth")      # Save the decomposed model weights.
        ```

    2. In the training script, call [decompose\_network](./en/api/decompose_network.md) to modify the model structure to the decomposed structure based on the decomposition information file obtained in [2.a](#li265516506413), then load the decomposed model weights saved in [2.a](#li265516506413) and perform fine-tuning.

        ```python
        from amct_pytorch.tensor_decompose import decompose_network
        net = Net()                                                                 # (*) Build the user model object.
        net, changes = decompose_network(                                           # Load the decomposition information file and modify the model structure to the tensor-decomposed structure.
            model=net,
            decompose_info_path="decomposed_path/decompose_info.json"               # Path of the decomposition information file saved in the previous step.
        )
        net.load_state_dict(torch.load("decomposed_path/decomposed_weights.pth"))   # Load the decomposed model weights saved in the previous step.
        optimizer = build_optimizer(net, ...)                                       # (*) Build the optimizer (pass the model parameters to the optimizer).
        train(net, optimizer, ...)                                                  # (*) fine-tune
        ```

</details>
