# accuracy\_based\_auto\_calibration<a name="ZH-CN_TOPIC_0000002517188734"></a>

## 产品支持情况<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |



## 功能说明<a name="zh-cn_topic_0240188739_section15406195619561"></a>

根据用户输入的模型、配置文件进行自动的校准过程，搜索得到一个满足目标精度的量化配置，输出可以在ONNX Runtime环境下做精度仿真的fake\_quant模型，和可在AI处理器上做推理的deploy模型。

## 函数原型<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section428121323411"></a>

```python
accuracy_based_auto_calibration(model,model_evaluator,config_file,record_file,save_dir,input_data,input_names,output_names,dynamic_axes,strategy='BinarySearch',sensitivity='CosineSimilarity')
```

## 参数说明<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240188739_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="11.87%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.34%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240188739_p1769255516412"><a name="zh-cn_topic_0240188739_p1769255516412"></a><a name="zh-cn_topic_0240188739_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="79.79%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240188739_p15231205416325"><a name="zh-cn_topic_0240188739_p15231205416325"></a><a name="zh-cn_topic_0240188739_p15231205416325"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240188739_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="11.87%" headers="mcps1.1.4.1.1 "><p id="p163296363467"><a name="p163296363467"></a><a name="p163296363467"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="8.34%" headers="mcps1.1.4.1.2 "><p id="p16326936194620"><a name="p16326936194620"></a><a name="p16326936194620"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.79%" headers="mcps1.1.4.1.3 "><p id="p813988817"><a name="p813988817"></a><a name="p813988817"></a>含义：用户的torch model。</p>
<p id="zh-cn_topic_0240188739_p11225740182619"><a name="zh-cn_topic_0240188739_p11225740182619"></a><a name="zh-cn_topic_0240188739_p11225740182619"></a>数据类型：torch.nn.Module</p>
</td>
</tr>
<tr id="row1763192415510"><td class="cellrowborder" valign="top" width="11.87%" headers="mcps1.1.4.1.1 "><p id="p1727017448554"><a name="p1727017448554"></a><a name="p1727017448554"></a>model_evaluator</p>
</td>
<td class="cellrowborder" valign="top" width="8.34%" headers="mcps1.1.4.1.2 "><p id="p19270154417552"><a name="p19270154417552"></a><a name="p19270154417552"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.79%" headers="mcps1.1.4.1.3 "><p id="p74219151189"><a name="p74219151189"></a><a name="p74219151189"></a>含义：自动量化进行校准和评估精度的Python实例。</p>
<p id="p227024455516"><a name="p227024455516"></a><a name="p227024455516"></a>数据类型：Python实例</p>
</td>
</tr>
<tr id="row147322675519"><td class="cellrowborder" valign="top" width="11.87%" headers="mcps1.1.4.1.1 "><p id="p112701544105513"><a name="p112701544105513"></a><a name="p112701544105513"></a>config_file</p>
</td>
<td class="cellrowborder" valign="top" width="8.34%" headers="mcps1.1.4.1.2 "><p id="p82709442556"><a name="p82709442556"></a><a name="p82709442556"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.79%" headers="mcps1.1.4.1.3 "><p id="p198671016389"><a name="p198671016389"></a><a name="p198671016389"></a>含义：用户生成的量化配置文件。</p>
<p id="p427074415514"><a name="p427074415514"></a><a name="p427074415514"></a>数据类型：string</p>
</td>
</tr>
<tr id="row581122875516"><td class="cellrowborder" valign="top" width="11.87%" headers="mcps1.1.4.1.1 "><p id="p6270174411554"><a name="p6270174411554"></a><a name="p6270174411554"></a>record_file</p>
</td>
<td class="cellrowborder" valign="top" width="8.34%" headers="mcps1.1.4.1.2 "><p id="p2270344115519"><a name="p2270344115519"></a><a name="p2270344115519"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.79%" headers="mcps1.1.4.1.3 "><p id="p187213181289"><a name="p187213181289"></a><a name="p187213181289"></a>含义：存储量化因子的路径，如果该路径下已存在文件，则会被重写。</p>
<p id="p22711144105518"><a name="p22711144105518"></a><a name="p22711144105518"></a>数据类型：string</p>
</td>
</tr>
<tr id="row18771230135514"><td class="cellrowborder" valign="top" width="11.87%" headers="mcps1.1.4.1.1 "><p id="p627114495515"><a name="p627114495515"></a><a name="p627114495515"></a>save_dir</p>
</td>
<td class="cellrowborder" valign="top" width="8.34%" headers="mcps1.1.4.1.2 "><p id="p8271184415512"><a name="p8271184415512"></a><a name="p8271184415512"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.79%" headers="mcps1.1.4.1.3 "><p id="p34055205820"><a name="p34055205820"></a><a name="p34055205820"></a>含义：模型存放路径。该路径需要包含模型名前缀，例如./quantized_model/*<em id="i2027174495514"><a name="i2027174495514"></a><a name="i2027174495514"></a>model</em>。</p>
<p id="p3271124435510"><a name="p3271124435510"></a><a name="p3271124435510"></a>数据类型：string</p>
</td>
</tr>
<tr id="row541516314356"><td class="cellrowborder" valign="top" width="11.87%" headers="mcps1.1.4.1.1 "><p id="p1041583193512"><a name="p1041583193512"></a><a name="p1041583193512"></a>input_data</p>
</td>
<td class="cellrowborder" valign="top" width="8.34%" headers="mcps1.1.4.1.2 "><p id="p44158315353"><a name="p44158315353"></a><a name="p44158315353"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.79%" headers="mcps1.1.4.1.3 "><p id="p18897721689"><a name="p18897721689"></a><a name="p18897721689"></a>含义：模型的输入数据。一个<span>torch.tensor</span><span>会被等价为</span><span>tuple</span><span>（</span><span>torch.tensor</span><span>）</span>。</p>
<p id="p7501837155915"><a name="p7501837155915"></a><a name="p7501837155915"></a>数据类型：tuple</p>
</td>
</tr>
<tr id="row1566581313515"><td class="cellrowborder" valign="top" width="11.87%" headers="mcps1.1.4.1.1 "><p id="p76651213193511"><a name="p76651213193511"></a><a name="p76651213193511"></a>input_names</p>
</td>
<td class="cellrowborder" valign="top" width="8.34%" headers="mcps1.1.4.1.2 "><p id="p1066501314355"><a name="p1066501314355"></a><a name="p1066501314355"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.79%" headers="mcps1.1.4.1.3 "><p id="p1761513238814"><a name="p1761513238814"></a><a name="p1761513238814"></a>含义：模型的输入的名称，用于modfied_onnx_file中显示。</p>
<p id="p196913313314"><a name="p196913313314"></a><a name="p196913313314"></a>默认值：None</p>
<p id="p15526359598"><a name="p15526359598"></a><a name="p15526359598"></a>数据类型：list(string)</p>
</td>
</tr>
<tr id="row10793191013517"><td class="cellrowborder" valign="top" width="11.87%" headers="mcps1.1.4.1.1 "><p id="p2079319106355"><a name="p2079319106355"></a><a name="p2079319106355"></a>output_names</p>
</td>
<td class="cellrowborder" valign="top" width="8.34%" headers="mcps1.1.4.1.2 "><p id="p17793111015359"><a name="p17793111015359"></a><a name="p17793111015359"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.79%" headers="mcps1.1.4.1.3 "><p id="p1280482514811"><a name="p1280482514811"></a><a name="p1280482514811"></a>含义：模型的输出的名称，用于modfied_onnx_file中显示。</p>
<p id="p5662141914377"><a name="p5662141914377"></a><a name="p5662141914377"></a>默认值：None</p>
<p id="p46621619183716"><a name="p46621619183716"></a><a name="p46621619183716"></a>数据类型：list(string)</p>
</td>
</tr>
<tr id="row70431133514"><td class="cellrowborder" valign="top" width="11.87%" headers="mcps1.1.4.1.1 "><p id="p149011843163518"><a name="p149011843163518"></a><a name="p149011843163518"></a>dynamic_axes</p>
</td>
<td class="cellrowborder" valign="top" width="8.34%" headers="mcps1.1.4.1.2 "><p id="p11093113355"><a name="p11093113355"></a><a name="p11093113355"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.79%" headers="mcps1.1.4.1.3 "><p id="p663519271087"><a name="p663519271087"></a><a name="p663519271087"></a>含义：对模型输入输出动态轴的指定，例如对于输入inputs（NCHW），N、H、W为不确定大小，输出outputs（NL），N为不确定大小，则{"inputs": [0,2,3], "outputs": [0]}。</p>
<p id="p58938131378"><a name="p58938131378"></a><a name="p58938131378"></a>默认值：None</p>
<p id="p1486351312114"><a name="p1486351312114"></a><a name="p1486351312114"></a>数据类型：dict&lt;string, dict&lt;python:int, string&gt;&gt; or dict&lt;string, list(int)<em id="i82727557411"><a name="i82727557411"></a><a name="i82727557411"></a>&gt;</em></p>
</td>
</tr>
<tr id="row19569102213553"><td class="cellrowborder" valign="top" width="11.87%" headers="mcps1.1.4.1.1 "><p id="p8271104455513"><a name="p8271104455513"></a><a name="p8271104455513"></a>strategy</p>
</td>
<td class="cellrowborder" valign="top" width="8.34%" headers="mcps1.1.4.1.2 "><p id="p727117445551"><a name="p727117445551"></a><a name="p727117445551"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.79%" headers="mcps1.1.4.1.3 "><p id="p778010297818"><a name="p778010297818"></a><a name="p778010297818"></a>含义：搜索满足精度要求的量化配置的策略，默认是二分法策略。</p>
<p id="p202712441551"><a name="p202712441551"></a><a name="p202712441551"></a>数据类型：string或Python实例</p>
<p id="p141261046574"><a name="p141261046574"></a><a name="p141261046574"></a>默认值：BinarySearch</p>
</td>
</tr>
<tr id="row262223511567"><td class="cellrowborder" valign="top" width="11.87%" headers="mcps1.1.4.1.1 "><p id="p1062310355567"><a name="p1062310355567"></a><a name="p1062310355567"></a>sensitivity</p>
</td>
<td class="cellrowborder" valign="top" width="8.34%" headers="mcps1.1.4.1.2 "><p id="p097993595712"><a name="p097993595712"></a><a name="p097993595712"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.79%" headers="mcps1.1.4.1.3 "><p id="p518211320810"><a name="p518211320810"></a><a name="p518211320810"></a>含义：评价每一层量化层对于量化敏感度的指标，默认是余弦相似度。</p>
<p id="p1797983525717"><a name="p1797983525717"></a><a name="p1797983525717"></a>数据类型：string或Python实例</p>
<p id="p67548209285"><a name="p67548209285"></a><a name="p67548209285"></a>默认值：CosineSimilarity</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

无

## 调用示例<a name="section179052217494"></a>

```python
import amct_pytorch as amct
from amct_pytorch.common.auto_calibration import AutoCalibrationEvaluatorBase

# You need to implement the AutoCalibrationEvaluator's calibration(), evaluate() and metric_eval() funcs
class AutoCalibrationEvaluator(AutoCalibrationEvaluatorBase):
    """ subclass of AutoCalibrationEvaluatorBase"""
    def __init__(self, target_loss, batch_num):
        super(AutoCalibrationEvaluator, self).__init__()
        self.target_loss = target_loss
        self.batch_num = batch_num

    def calibration(self, model):
        """ implement the calibration function of AutoCalibrationEvaluatorBase
            calibration() need to finish the calibration inference procedure
            so the inference batch num need to >= the batch_num pass to create_quant_config
        """
        model_forward(model=model, batch_size=32, iterations=self.batch_num)

    def evaluate(self, model):
        """ implement the evaluate function of AutoCalibrationEvaluatorBase
            params: model in torch.nn.module 
            return: the accuracy of input model on the eval dataset, or other metric which
                    can describe the 'accuracy' of model
        """
        top1, _ = model_forward(model=model, batch_size=32, iterations=5)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return top1

    def metric_eval(self, original_metric, new_metric):
        """ implement the metric_eval function of AutoCalibrationEvaluatorBase
            params: original_metric: the returned accuracy of evaluate() on non quantized model
                    new_metric: the returned accuracy of evaluate() on fake quant model
            return:
                   [0]: whether the accuracy loss between non quantized model and fake quant model
                        can satisfy the requirement
                   [1]: the accuracy loss between non quantized model and fake quant model
        """
        loss = original_metric - new_metric
        if loss * 100 < self.target_loss:
            return True, loss
        return False, loss
    ...
    # 1. step1 create quant config json file
    config_json_file = os.path.join(TMP, 'config.json')
    skip_layers = []
    batch_num = 2
    amct.create_quant_config(
        config_json_file,
        model,
        input_data,
        skip_layers,
        batch_num
    )

    # 2. step2 construct the instance of AutoCalibrationEvaluator
    evaluator = AutoCalibrationEvaluator(target_loss=0.5, batch_num=batch_num)

    # 3. step3 using the accuracy_based_auto_calibration to quantized the model
    record_file = os.path.join(TMP, 'scale_offset_record.txt')
    result_path = os.path.join(PATH, 'result/mobilenet_v2')
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

落盘文件说明：

-   精度仿真模型文件：ONNX格式的模型文件，模型名中包含fake\_quant，可以在ONNX Runtime环境进行精度仿真。
-   部署模型文件：ONNX格式的模型文件，模型名中包含deploy，经过ATC转换工具转换后可部署到AI处理器。
-   量化因子记录文件：在接口中的record\_file中写入量化因子。
-   敏感度信息文件：该文件记录了待量化层对于量化的敏感度信息，根据该信息进行量化回退层的选择。
-   自动量化回退历史记录文件：记录的回退层的信息。

