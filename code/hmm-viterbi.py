import csv

file = open("../metu_train.txt", "r",  encoding="utf8")
file2 = open("../metu_test.txt", "r",  encoding="utf8")

def diff(list1, list2):
    list2 = set(list2)
    return [item for item in list1 if item not in list2]


lines = file.read().split("\n")
lines2 = file2.read().split("\n")

# -------------------- TASK 1: BUILD A HMM -------------------- #
def get_frequency(tag_dict):
    result = 0

    for tag, frequency in tag_dict.items():
        result += frequency

    return result


initial_frequency = {}
initial_probability = {}
transition_frequency = {}
transition_probability = {}
emission_frequency = {}
emission_probability = {}

tags_dict = {}

all_tags = []
all_words = []

# #### INITIAL FREQUENCY AND READ TRAIN DATA #####
for sentence in lines:
    wordTag = sentence.split(" ")
    tags = []
    words = []

    initial_frequency[wordTag[0].split("/")[1]] = initial_frequency.get(wordTag[0].split("/")[1], 0) + 1

    for i in range(len(wordTag)):
        wtpair = wordTag[i].split("/")
        tags.append(wtpair[1])
        words.append(wtpair[0].lower())
        tags_dict[wtpair[1]] = tags_dict.get(wtpair[1], 0) + 1

    for tag in tags:
        all_tags.append(tag)

    for word in words:
        all_words.append(word)

start_size = get_frequency(initial_frequency)

# #### INITIAL PROBABILITY #####
for tag, freq in tags_dict.items():
    if tag in initial_frequency.keys():
        initial_probability[tag] = initial_frequency[tag] / start_size
    else:
        initial_probability[tag] = 0.0 / start_size

for i in range(len(all_tags)):
    transition_frequency[all_tags[i]] = {}
    transition_probability[all_tags[i]] = {}
    emission_frequency[all_tags[i]] = {}
    emission_probability[all_tags[i]] = {}

# #### TRANSITION FREQUENCY AND PROBABILITY #####
for i in range(len(lines)):
    wordTag = lines[i].split(" ")

    for j in range(len(wordTag) - 1):
        pair = wordTag[j].split("/")

        for k, v in transition_frequency.items():
            if k == pair[1]:
                transition_frequency[k].update({wordTag[j+1].split("/")[1]: transition_frequency[k].get(
                    wordTag[j+1].split("/")[1], 0) + 1})

for k, v in transition_frequency.items():
    sum = get_frequency(v)
    for k2, v2 in v.items():
        transition_probability[k][k2] = v2 / sum

# #### EMISSION FREQUENCY AND PROBABILITY #####
for i in range(len(all_words)):
    for k, v in emission_frequency.items():
        if k == all_tags[i]:
            emission_frequency[k].update({all_words[i]: emission_frequency[k].get(all_words[i], 0) + 1})

for k, v in emission_frequency.items():
    sum = get_frequency(v)
    for k2, v2 in v.items():
        emission_probability[k][k2] = v2 / sum


# -------------------- TASK 2: VITERBI ALGORITHM -------------------- #
def get_once_word(word_dict):
    count = 0
    for k, v in word_dict.items():
        if v == 1:
            count += 1
    return count


all_test_words = []

once_word_of_tags = {}

# #### READ TEST DATA #####
for sentence in lines2:
    wordTag = sentence.split(" ")
    tags = []
    words = []

    for i in range(len(wordTag)):
        wtpair = wordTag[i].split("/")
        tags.append(wtpair[1])
        words.append(wtpair[0].lower())

    for word in words:
        all_test_words.append(word)


# #### NUMBER OF WORDS WHICH APPEARED ONCE WITH TAG T #####
for tag in emission_frequency.keys():
    once_word_of_tags[tag] = get_once_word(emission_frequency[tag])


# #### NUMBER OF UNKNOWN WORDS WHICH APPEARED IN TEST DATA #####
unk_words = len(diff(all_test_words, all_words))


# #### VITERBI ALGORITHM #####
def viterbi(observation, states, one_freq_words, start_prob, transition_prob, emission_prob):
    viterbi_matrix = {}
    backpointer = {}
    observation_length = len(observation)

    # Initial word given
    for tag in states:
        first_word = observation[0]
        if first_word in emission_prob[tag].keys():
            viterbi_matrix[0, tag] = start_prob[tag] * emission_prob[tag].get(first_word, 0)
            backpointer[0, tag] = None
        else:
            viterbi_matrix[0, tag] = start_prob[tag] * (one_freq_words[tag] / (unk_words * get_frequency(emission_frequency[tag])))
            backpointer[0, tag] = None

    # Examine the second word now, we already examined first word
    for time_step, word in enumerate(observation[1:], start=1):
        for tag in states:
            if word in emission_prob[tag].keys():
                p_emission = emission_prob[tag].get(word, 0)
            else:
                p_emission = one_freq_words[tag] / (unk_words * get_frequency(emission_frequency[tag]))

            # argmax for the viterbi and backpointer
            probability, state = max((viterbi_matrix[time_step - 1, prev_tag] * transition_prob[prev_tag].get(tag, 0),
                                          prev_tag) for prev_tag in tags)

            # max probability for the viterbi matrix
            viterbi_matrix[time_step, tag] = probability * p_emission

            # state of the max probability for the backpointer
            backpointer[time_step, tag] = state

    # Termination steps
    probability, state = max((viterbi_matrix[observation_length - 1, tag] * transition_prob[tag].get('Punc', 0), tag)
                             for tag in tags)
    viterbi_matrix[observation_length, 'Punc'] = probability
    backpointer[observation_length, 'Punc'] = state

    # Return backtrace path...
    backtrace_path = []
    previous_state = 'Punc'
    for index in range(observation_length, 0, -1):
        state = backpointer[index, previous_state]
        backtrace_path.append(state)
        previous_state = state

    # We are tracing back through the pointers, so the path is in reverse
    backtrace_path = list(reversed(backtrace_path))
    return backtrace_path


# -------------------- TASK 3: EVALUATION -------------------- #
def calculate_accuracy(real_tags, test_tags):
    total_tags = 0
    true = 0
    for i in range(len(test_tags)):
        if test_tags[i] == real_tags[i]:
            true += 1
        total_tags += 1
    return true, total_tags


# #### CHECK THE TAGS IN TEST SENTENCES #####
f = open("output.txt", "w", encoding='utf-8')

#csvData = []
#csvData.append(['Id', 'Category'])

acc_total = 0
acc_correct = 0
#total_tags = 0
for sentence in lines2:
    wordTag = sentence.split(" ")
    tags = []
    words = []

    actual_words = [sentence.split(" ")[i].split("/")[0] for i in range(len(wordTag))]

    string = ""
    for i in range(len(wordTag)):
        wtpair = wordTag[i].split("/")
        tags.append(wtpair[1])
        words.append(wtpair[0].lower())

    viterbi_result = viterbi(words, tags_dict.keys(), once_word_of_tags, initial_probability, transition_probability,
                             emission_probability)

    string = ""
    for i in range(len(actual_words)):
        #kaggle = []
        #total_tags += 1
        string += actual_words[i] + "/" + viterbi_result[i] + " "
        #kaggle.append(str(total_tags))
        #kaggle.append(viterbi_result[i])
        #csvData.append(kaggle)

    string += "\n"

    f.write(string)

    true, total = calculate_accuracy(tags, viterbi_result)

    acc_total += total
    acc_correct += true

string = "Accuracy: " + str((acc_correct / acc_total) * 100)
print("Correct found tags:", acc_correct)
print("Total words:", acc_total)
print(string)

# with open('output.csv', 'w', newline='') as csvFile:
#     writer = csv.writer(csvFile)
#     writer.writerows(csvData)
#
# csvFile.close()

file.close()
f.close()
