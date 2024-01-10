import torch
import torch.nn as nn


def rnn_output_test(
    our_rnn,
    pytorch_rnn,
    x,
    val=0.3,
    tol=1e-9
):

    for p in pytorch_rnn.parameters():
        nn.init.constant_(p, val=val)
    for p in our_rnn.parameters():
        nn.init.constant_(p, val=val)

    output_pytorch, h_pytorch = pytorch_rnn(x)
    output_our, h_our = our_rnn(x)

    if isinstance(h_pytorch, tuple):
        assert isinstance(h_our, tuple) and len(h_pytorch) == len(h_our), \
            'Different rnn types {} vs {}!'.format(
                our_rnn.__class__.__name__, pytorch_rnn.__class__.__name__
            )
        h_pytorch, c_pytorch = h_pytorch
        h_our, c_our = h_our
    else:
        c_pytorch, c_our = None, None

    # Outputs must have the same shapes
    passed = True
    if output_pytorch.data.shape == output_our.data.shape:
        print('Output shape test passed, {} == {}'.format(
            output_pytorch.data.shape, output_our.data.shape
        ))
    else:
        print('Output shape test failed, {} != {}'.format(
            output_pytorch.shape, output_our.shape
        ))
        passed = False
    if h_pytorch.shape == h_our.shape:
        print('Hidden shape test passed, {} == {}'.format(
            h_pytorch.shape, h_our.shape
        ))
    else:
        print('Hidden shape test failed, {} != {}'.format(
            h_pytorch.shape, h_our.shape
        ))
        passed = False

    for output, name in zip(
        [(output_our, output_pytorch), (h_our, h_pytorch), (c_our, c_pytorch)],
        ['h_seq', 'h', 'c']
    ):
        if output[0] is None or output[1] is None:
            continue

        if not passed:
            print('Your model has some shape mismatches, check your implementation.')
        else:
            # The difference of outputs should be 0
            diff = torch.sum((output[0].data - output[1].data)**2)
            print("\nDifference between pytorch and your RNN implementation for '{}': {:.2f}".format(
                name, diff.item()
            ))
            if diff.item() < tol:
                print("Congrats, you implemented a correct model.")
            else:
                print("There is something wrong in your model. Try again.")
                passed = False
                break

    return passed


def embedding_output_test(
    our_embedding,
    pytorch_embedding,
    x,
    val=0.3,
    tol=1e-9
):
    our_embedding.weight.data.copy_(pytorch_embedding.weight.data)

    our_output = our_embedding(x)
    torch_output = pytorch_embedding(x)
    passed = True
    if our_output.shape != torch_output.shape:
        passed = False
        print('Output shapes are mismatched. {} vs {}'.format(
            our_output.shape, torch_output.shape
        ))
        
    if not our_embedding(x).requires_grad:
        
        print('Warning: Your embeddings are not trainable. Check your implementation.')
        
    if passed:
        diff = (our_output - torch_output).pow(2).sum().sqrt().item()
        print('Difference between outputs: {}'.format(diff))

        if diff < 1e-9:
            print('Test passed.')
        else:
            print('Test failed, check your implementation.')
            passed = False

    return passed


def classifier_test(classifier, num_embeddings):
    # Define some constants
    seq_len=10
    batch_size=3
    
    # Create a random sequence
    x = torch.randint(0, num_embeddings-1, (seq_len, batch_size))

    # Test the output format
    y = classifier(x)
    passed = True
    if not torch.logical_and((y <= 1), (y >= 0)).all():
        print('Your model does not output probabilities between 0 and 1.')
        passed = False
    if y.shape != (batch_size, ):
        print('Your model does not produce a 1-D output of shape (batch_size, )')
        passed = False

    # Test varying batch sizes
    assert seq_len-batch_size > 0, "Seq len must be bigger than batch size"
    lengths = torch.tensor([seq_len-i for i in range(batch_size)]).long()
    batched_outputs = classifier(x, lengths)
    regular_outputs = torch.stack([
        classifier(x[:lengths[i], i].unsqueeze(1))
        for i in range(lengths.numel())
    ]).squeeze()

    if batched_outputs.shape != regular_outputs.shape:
        print('Output with lengths {} produced wrong size argument {} vs {}'.format(
            lengths.tolist(), batched_outputs.shape, regular_outputs.shape
        ))
        print('Make sure you handle lengths argument properly in your classifier.')
        passed = False

    diff = torch.norm(batched_outputs - regular_outputs)
    if diff > 1e-9:
        print('Output with lengths {} has a large error: {}'.format(lengths.tolist(), diff))
        print('Make sure you handle lengths argument properly in your classifier.') 
        passed = False

    # Log the final result
    if passed:
        print('All output tests are passed!')
    else:
        print('Some output tests are failed!')
    return passed


def parameter_test(model):
    total = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: {}'.format(total))
    if total < 2 * 1e6:
        print('Your model is sufficiently small.')
        return True
    else:
        print('Your model is too large. Shrink its size.')
        return False
