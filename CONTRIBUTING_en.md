# Contributing Guide

This project welcomes developers to experience and contribute. Before participating in community contributions, please refer to [cann-community](https://gitcode.com/cann/community) to understand the code of conduct, sign the CLA agreement, and learn about the contribution process of the source repository.

Developers need to pay attention to the following points when preparing local code and submitting PRs:

1. When submitting a PR, please carefully fill in the business background, purpose, and solution information according to the PR template.

2. If your modification is not a simple bug fix but involves new features, new interfaces, new configuration parameters, or modifications to the code flow, please discuss the solution through an Issue first to avoid your code being rejected. If you are unsure whether your modification can be classified as a "simple bug fix," you can also discuss the solution by submitting an Issue.

3. Please use the `pre-commit` tool to ensure the code meets basic requirements. After installation, it will automatically check the code style of the current submission during git commit.

   ```shell
   pip install pre-commit
   pre-commit install
   ```

Developer contribution scenarios mainly include:

## Bug Fix

If you discover certain bugs in this project and wish to fix them, you are welcome to create an Issue for feedback and tracking.

You can follow the [Submit Issue/Process Issue Task](https://gitcode.com/cann/community#%E6%8F%90%E4%BA%A4Issue%E5%A4%84%E7%90%86Issue%E4%BB%BB%E5%8A%A1) guide to create a `Bug-Report|Defect Feedback` type Issue to describe the bug. Then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself for processing.

## Contribute New Features

Thank you for participating in AMCT development and adding more value to cann-amct! To make your contribution process smoother and your solution better aligned with community needs, here are detailed guidelines for your reference:

### Submit RFC to Reach Consensus on Solution
Before formal development, we recommend that you submit an RFC (Request for Comments document, submitted by creating an ISSUE) first to fully exchange solution details with other developers and reach consensus. Please include at least the following in the RFC:
- Background and motivation of the feature: For example, what scenario requirements this feature addresses.
- Core design solution: Briefly explain the technical approach for implementing this feature and the design solution of key modules.
- Expected functional and performance goals: For example, the core functions this feature needs to implement and the expected accuracy metrics or performance.
- Estimated completion time: Helps the community understand development progress and facilitates subsequent collaborative support.

### Submit PR to Complete Delivery Content
After the solution consensus is reached, you can submit the corresponding PR and remember to link it to the previous RFC. To ensure sample quality and reusability, please include at least the following in the PR:
- Feature code: Please ensure the code style meets [community basic requirements](https://gitcode.com/cann/community/tree/master/contributor/coding-standards) and can pass the pipeline code check, making it easier for subsequent developers to understand and use.
- Optimization documentation: Please explain in detail the key content during functional adaptation and performance optimization, such as "Why did we make this optimization point," "What method was used to implement it," and "What specific benefits did the optimization bring (such as performance improvement by X%)."
- README documentation: This is key to helping other developers get started quickly and needs to include two parts:
    - Brief description: The core content of the current feature, such as the model used, supported execution devices, verified accuracy/performance information, etc.
    - Operation steps: Please describe the full process from environment preparation to feature execution in as much detail as possible to ensure other developers can smoothly reproduce results.

### Some Small Reminders to Make Contribution Smoother
To avoid small setbacks during subsequent integration, here are a few details we want to share with you:
- Except for images needed in README or documentation, please do not include binary files in the submitted code.
- If feature verification involves the use of third-party datasets, just explain the download method and usage method of the dataset in the documentation; you do not need to provide the dataset directly.
- If your modification involves public code (not internal code of this feature), you need to ensure it can pass CI validation.
- If feature development involves the addition or modification of operators, please first merge the corresponding operator changes into the operator repository, then proceed with the compression feature integration.
- Please check whether the LICENSE you use is compliant. We recommend using Apache 2.0 and other agreements and marking copyright information according to actual conditions.

### Directory Structure Reference
If the feature you contribute has not undergone strong generalization testing, we recommend merging the feature code into the experimental directory (such as amct_pytorch/experimental). You can refer to the following structure to organize your code and documentation:
```
├── experimental                             # Experimental features
|  ├── sample1                               # Feature name (such as custom_defined_quantize_alg)
|  |   ├── doc                               # doc directory: store optimization documents, images, etc.
|  |   ├── src                               # src directory: store code
|  |   ├── README.md                         # The README documentation mentioned above
|  |   └── ...                               # Other necessary files (such as environment configuration files, etc.)
|  ├── sample2
│  └── ...
```

If you have any questions during the contribution process, feel free to communicate in the community at any time. Thank you again for your support, and we look forward to your wonderful contributions!

## Documentation Correction

If you discover certain documentation description errors in this project, you are welcome to create an Issue for feedback and correction.

You can follow the [Submit Issue/Process Issue Task](https://gitcode.com/cann/community#%E6%8F%90%E4%BA%A4Issue%E5%A4%84%E7%90%86Issue%E4%BB%BB%E5%8A%A1) guide to create a `Documentation|Documentation Feedback` type Issue to point out the problems in the corresponding documentation. Then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself for correcting the corresponding documentation description.

## Help Solve Others' Issues

If you have suitable solutions for problems encountered by others in the community, you are welcome to publish comments in the Issue to communicate and help others solve problems and pain points, jointly optimizing usability.

If the corresponding Issue requires code modification, you can enter "/assign" or "/assign @yourself" in the Issue comment box to assign the Issue to yourself and track assistance in solving the problem.