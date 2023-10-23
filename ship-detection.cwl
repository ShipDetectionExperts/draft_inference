cwlVersion: v1.0
$graph:
  - id: ship-detection-workflow
    class: Workflow
    label: inferenceworkflow
    inputs:
      - id: client_id
        type: string
      - id: client_secret
        type: string
      - id: bbox
        type: string
      - id: time
        type: string
      - id: maxcc
        type: int
      - id: threshold
        type: float
    outputs:
      - id: wf_outputs
        outputSource:
          - ship-inference_step/results
        type: Directory
    steps:
      - id: ship-inference_step
        in:
          - id: client_id
            source:
              - client_id
          - id: client_secret
            source:
              - client_secret
          - id: bbox
            source:
              - bbox
          - id: time
            source:
              - time
          - id: maxcc
            source:
              - maxcc
          - id: threshold
            source:
              - threshold
        out:
          - results
        run: '#ship-inference'
  - id: ship-inference
    class: CommandLineTool
    baseCommand:
      - python
      - /app/src/inference.py
    label: perform inference
    doc: Run the inference script with input parameters
    inputs:
      - id: client_id
        type: string
        inputBinding:
          prefix: '--client_id'
      - id: client_secret
        type: string
        inputBinding:
          prefix: '--client_secret'
      - id: bbox
        type: string
        inputBinding:
          prefix: '--bbox'
      - id: time
        type: string
        inputBinding:
          prefix: '--time'
      - id: maxcc
        type: int
        inputBinding:
          prefix: '--maxcc'
      - id: threshold
        type: float
        inputBinding:
          prefix: '--threshold'
    outputs:
      - id: results
        type: Directory
        outputBinding:
          glob: .
    requirements:
      DockerRequirement:
        dockerPull: potato55/ship-detection:0.1
$namespaces:
  s: https://schema.org/
s:softwareVersion: 0.0.1
s:dateCreated: '2023-10-23'
s:keywords: ship detection, machine learning, ap
s:contributor:
  - s:name: Selim Behloul
    s:email: ADDSELIMEMAIL
    s:affiliation: 'Colleague '
s:codeRepository: https://github.com/ShipDetectionExperts/draft_inference/tree/application-package
s:releaseNotes: none
s:license: none
s:author:
  - s:name: Juraj Zvolensky
    s:email: juro.zvolensky@gmail.com
    s:affiliation: AP Enjoyer