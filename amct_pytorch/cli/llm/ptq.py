from amct_pytorch.cli.llm.args import parser_gen
from amct_pytorch.workflows.llm_ptq import LlmPtqWorkflow


def main():
    args = parser_gen(command="ptq")
    workflow = LlmPtqWorkflow(args)
    workflow.run()

if __name__ == "__main__":
    main()
