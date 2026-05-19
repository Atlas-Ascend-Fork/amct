from amct_pytorch.cli.llm.args import parser_gen
from amct_pytorch.workflows.llm_eval import LlmEvalWorkflow


def main():
    args = parser_gen(command="eval")
    workflow = LlmEvalWorkflow(args)
    workflow.run()

if __name__ == "__main__":
    main()
