from amct_pytorch.cli.llm.args import parser_gen
from amct_pytorch.workflows.llm_deploy import LlmDeployWorkflow


def main():
    args = parser_gen(command="deploy")
    workflow = LlmDeployWorkflow(args)
    workflow.run()

if __name__ == "__main__":
    main()
