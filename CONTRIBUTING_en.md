# Contributing Guide

This project welcomes developers to experience and participate in contributions. Before participating in community contributions, refer to [cann-community](https://gitcode.com/cann/community) to understand the code of conduct, sign the CLA agreement, and learn about the contribution process of the source code repository.

Developers need to pay attention to the following points when preparing local code and submitting PRs:

1. When submitting a PR, carefully fill in the business background, purpose, solution, and other information according to the PR template.

2. If your modification is not a simple bug fix but involves new features, new interfaces, new configuration parameters, or modifications to code flow, discuss the solution through an Issue first to avoid your code being rejected. If you are unsure whether your modification can be classified as a "simple bug fix," you can also submit an Issue for solution discussion.

3. Use the `pre-commit` tool to ensure the code meets basic requirements. After installation, it will automatically check code compliance issues during git commit.

    ```shell
    pip install pre-commit
    pre-commit install
    ```


Developer contribution scenarios mainly include:

## Bug Fix

If you discover certain bugs in this project and wish to fix them, you are welcome to create a new Issue for feedback and tracking.

Follow the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#%E6%8F%90%E4%BA%A4Issue%E5%A4%84%E7%90%86Issue%E4%BB%BB%E5%8A%A1) guide to create a `Bug-Report|缺陷反馈` type Issue to describe the bug, then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself for processing.

## Contributing New Features

Thank you for participating in AMCT development and injecting more value into cann-amct! To make your contribution process smoother and ensure your solution better fits community needs, the following detailed operation guide is for your reference:

### Submit RFC, Reach Solution Consensus
Before formal development, it is recommended that you submit an RFC (Request for Comments document, submitted by creating an ISSUE) first to fully exchange solution details with other developers and reach consensus. The RFC should at least include the following:
- Feature background and motivation: For example, what scenario needs the feature addresses.
- Core design solution: Briefly explain the technical approach of the feature implementation and the design solution of key modules.
- Expected functional and performance goals: For example, the core functions the feature needs to implement, expected precision metrics, or performance results.
- Estimated completion time: To help the community understand development progress and facilitate subsequent collaborative support.

### Submit PR, Improve Delivery Content
After reaching solution consensus, you can submit the corresponding PR, and remember to link it to the previous RFC. To ensure sample quality and reusability, the PR should at least include the following:
- Feature code: Ensure the code style complies with [community basic requirements](https://gitcode.com/cann/community/tree/master/contributor/coding-standards) and can pass the pipeline code check, making it easier for subsequent developers to understand and use.
- Optimization documentation: Describe in detail the key content of the feature in terms of functional adaptation and performance optimization, such as "why this optimization point was needed," "what method was used to implement it," and "what specific benefits the optimization brought (such as X% performance improvement)."
- README documentation: This is key to helping other developers get started quickly and needs to include two parts:
    - Brief introduction: The core content of the current feature, such as the model used, supported execution devices, verified precision/performance information, and so on.
    - Operation steps: Describe in as much detail as possible the complete process from environment preparation to feature execution, ensuring other developers can smoothly reproduce results.

### Some Reminders for Smoother Contributions
To avoid minor setbacks during subsequent merging, here are a few small details to share with you:
- Except for images needed in README or documentation, do not include binary files in the merged code.
- If feature verification involves the use of third-party datasets, only need to describe the dataset download method and usage method in the documentation, without directly providing the dataset.
- If your modification involves public code (not internal code of this feature), ensure it can pass CI verification.
- If feature development involves operator addition or modification, merge the corresponding operator changes into the operator repository first, then proceed with compression feature merging.
- Check whether the LICENSE you use is compliant; it is recommended to use Apache 2.0 or similar agreements, and mark copyright information according to actual conditions.

### Directory Structure Reference
If the feature you contribute has not undergone strong generalization testing, it is recommended to merge the feature code into the experimental directory (such as amct_pytorch/experimental). You can refer to the following structure to organize your code and documentation:
```
├── experimental                             # Experimental features
|  ├── sample1                               # Feature name (such as custom_defined_quantize_alg)
|  |   ├── doc                               # doc directory: store optimization documentation, images, etc.
|  |   ├── src                               # src directory: store code
|  |   ├── README.md                         # README documentation mentioned above
|  |   └── ...                               # Other necessary files (such as environment configuration files, etc.)
|  ├── sample2
│  └── ...
```

If you have any questions during the contribution process, feel free to communicate in the community at any time; thank you again for your support, and we look forward to your wonderful contributions!

## Documentation Correction

If you discover certain documentation description errors in this project, you are welcome to create a new Issue for feedback and correction.

Follow the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#%E6%8F%90%E4%BA%A4Issue%E5%A4%84%E7%90%86Issue%E4%BB%BB%E5%8A%A1) guide to create a `Documentation|文档反馈` type Issue to point out the corresponding documentation problem, then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself for correcting the corresponding documentation description.

## Help Solve Others' Issues

If you have suitable solutions for problems others encounter in the community, you are welcome to publish comments in the Issue for exchange, help others solve problems and pain points, and jointly optimize usability.

If the corresponding Issue requires code modification, you can enter "/assign" or "/assign @yourself" in the Issue comment box to assign the Issue to yourself and track assistance in solving the problem.