cwlVersion: v1.0
$graph:
  - id: hello-world-app
    class: Workflow
    inputs:
      - id: param
        type: string
    outputs:
      - id: wf_outputs
        outputSource:
          - hello_world/results
        type: Directory
    steps:
      - id: hello_world
        in:
          - id: param
            source:
              - param
        out:
          - results
        run: '#hello_world'
  - id: hello_world
    class: CommandLineTool
    baseCommand:
      - python
      - /app/hello_world.py
    inputs:
      - id: param
        type: string
        inputBinding:
          prefix: '--param'
    outputs:
      - id: results
        type: Directory
        outputBinding:
          glob: .
    requirements:
      DockerRequirement:
        dockerPull: potato55/hello-world-app:7.0
$namespaces:
  s: https://schema.org/
s:softwareVersion: 1.0.0
s:dateCreated: '2023-10-10'
s:codeRepository: https://github.com/jzvolensky/sentinel1-app-package/tree/experimental/src-exp
s:license: none
s:author:
  - s:name: Juraj Zvolensky
    s:email: email@email.com