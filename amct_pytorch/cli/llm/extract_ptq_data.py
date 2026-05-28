from amct_pytorch.cli.llm.args import parser_gen
from amct_pytorch.workflows.llm_extract_ptq_data import LlmExtractPtqDataWorkflow


def main():
    args = parser_gen(command="extract_ptq_data")
    workflow = LlmExtractPtqDataWorkflow(args)
    workflow.run()

if __name__ == "__main__":
    main()
