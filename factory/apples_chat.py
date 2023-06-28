from sentence_transformers import SentenceTransformer, util
import torch
import random
model = SentenceTransformer('all-MiniLM-L12-v2')


from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L12-v2')

# get the query from the user
query_input = input('What is your question?')

# encode the query using the model
query_embedding = model.encode(query_input)
passage_embedding = model.encode(['How did the batch turn out?',
                                  'was the batch accepted?',
                                  'Was the batch rejected',
                                  'what quality class is this batch?',
                                  'The quality of this batch is very bad, why?',
                                  'Why was this batch rejected?',
                                  'How many apples were rejected?',
                                  'How many apples were tested?',
                                  'How many apples were good?',
                                  'How many apples were bad?',
                                  'Do apples smell bad?'])

# get the similarity scores of first query with all other queries
similarity_scores = util.dot_score(query_embedding, passage_embedding)
softmax_scores = torch.softmax(similarity_scores, dim=1)

# Get the label of the most appropriate result
max_score, max_index = torch.max(softmax_scores, dim=1)
label = max_index.item()

# # labeling test prints
# print("Similarity:", similarity_scores)
# print("Softmax scores:", softmax_scores)
print(f'The most apropriate label for the question: {query_input} is: {label}')



random = random.randint(0, 3)
AQL_label = [random,1,22,0,9, 500, 32]


def query_answer(label, test_results):
    test_Klasse = test_results[0]
    test_apple_rot = test_results[1]
    test_appel_normal = test_results[2]
    test_apple_blotch = test_results[3]
    test_apple_spot = test_results[4]
    test_lot_size = test_results[5]
    test_batch_size = test_results[6]
    
    class_roman = {0: 'I', 1: 'II', 2: 'III'}

    def reject_norm(test_Klasse):
        if test_Klasse >= 2:
            return 'The batch was accepted'
        return 'The batch was rejected'

    if label == 0:
        return f'{reject_norm(test_Klasse)} with Class_{class_roman[test_Klasse]}'
    elif label == 1 or label == 2:
        return reject_norm(test_Klasse)
    elif label == 3:
        return f'The batch was assigned Class_{class_roman[test_Klasse]}'
    elif label == 4:
        return f"Too many apples {test_apple_rot+test_apple_blotch+test_apple_spot} were rejected, batch is deemed unfit to be processed in human foostuffs"
    elif label == 5:
        return f'Of the batch of {test_lot_size} apples, {test_batch_size} were tested. {test_apple_rot} apples were rejected because of rot, {test_apple_blotch} apples were rejected because of blotch, {test_apple_spot} apples were rejected because of spot'
    elif label == 6 or label == 9:
        return f' {test_batch_size} apples were tested {test_batch_size-test_appel_normal} apples were rejected because of rot, blotch or spot, {test_appel_normal} apples were good'
    elif label == 7:
        return f' {test_batch_size} apples were tested of the lot of {test_lot_size} apples'
    elif label == 8:
        return f'Of the lot of {test_batch_size}, {test_appel_normal} apples were good'
    elif label == 10:
        return "Apples don't smell bad, unless they are bad!"   # This is a joke by my wife
    else:
        return 'I am sorry, I do not understand your question'
    
print(query_input)
print(query_answer(label, AQL_label))

