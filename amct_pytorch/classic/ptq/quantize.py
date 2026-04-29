__all__ = []
from amct_pytorch.algorithms.registry_factory import QUANT_ALGO_REGISTRY
from amct_pytorch.common.optimization.global_solver import GlobalSolver

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from torch import nn

from amct_pytorch.algorithms import AlgorithmRegistry
from amct_pytorch.common.config import set_default_config, parse_config
from amct_pytorch.common.utils.check_params import check_params
from amct_pytorch.classic.ptq.optimizer import (
    InsertQuantizeModulePass,
    ModelOptimizer,
    ReplaceNpuQuantModulePass,
)
from amct_pytorch.classic.ptq.quantize_op.base_quant_module import BaseQuantizeModule


__all__ = ['quantize', 'convert', 'algorithm_register']


@check_params(model=nn.Module,
              config=(dict, type(None)))
def quantize(model, config=None):
    """
    Function: Modify user's model for quantization
    Parameter: model: user mode instance of Torch.nn.Module
               config: simply config dict from user to set
    Return: None
    """
    if config is None:
        config = set_default_config()

    layer_config = parse_config(model, config, AlgorithmRegistry)

    optimizer = ModelOptimizer()
    optimizer.add_pass(InsertQuantizeModulePass(layer_config))
    optimizer.do_optimizer(model)


@check_params(model=nn.Module)
def convert(model):
    """
    Function: Convert quantized calibration model to quantized deployment model
    Parameter: model: quantized calibration model instance of Torch.nn.Module
    Return: None
    """
    optimizer = ModelOptimizer()
    optimizer.add_pass(ReplaceNpuQuantModulePass())
    optimizer.do_optimizer(model)


@check_params(name=str,
              src_op=str,
              quant_op=BaseQuantizeModule,
              deploy_op=(nn.Module, type(None)))
def algorithm_register(name, src_op, quant_op, deploy_op=None):
    """
    Function: quantization algorithm registration
    Parameter: name: str, algorithm's name
               src_op: str, Source operators corresponding to the algorithm
               quant_op: nn.module or type, Quantized operators corresponding to the algorithm
               deploy_op: nn.module or type, Deployment operators corresponding to the algorithm
    Return: None
    """
    AlgorithmRegistry.register(name, src_op, quant_op, deploy_op=deploy_op)
