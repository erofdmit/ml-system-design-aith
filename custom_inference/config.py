class Opt:
    imgH = 32
    imgW = 100
    input_channel = 1
    batch_max_length = 25
    character = "0123456789"
    Transformation = "TPS"
    FeatureExtraction = "ResNet"
    SequenceModeling = "BiLSTM"
    Prediction = "Attn"
    num_fiducial = 20
    output_channel = 512
    hidden_size = 256
    num_class = len(character) + 2
