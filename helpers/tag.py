import sys

def zeroContext(all_input_lines):
    new_input_lines = []
    lines_record = []

    for line in all_input_lines:
        line = line.strip().split()
        lines_record.append(str(len(line)))
        for word in line:
            new_input_lines.append("<bow> " + word + " <eow>")

    return new_input_lines, lines_record

def nonZeroContext(all_input_lines, context):
    new_input_lines = []
    lines_record = []

    # Making tagged version of arabizi file
    for line in all_input_lines:
        line = line.strip().split()

        lines_record.append(str(len(line)))

        line = (["<bos>"] * context) + line  + (["<eos>"] * context)
        for word in range(context, len(line) - context):
            newLine = line[word - context: word] + ["<bow>", line[word], "<eow>"] + line[word + 1: word + context + 1]
            strLine = " ".join(newLine)
            new_input_lines.append(strLine)

    return new_input_lines, lines_record

def tag(all_input_lines, all_output_lines, context, mode):
    if context == 0:
        all_input_lines, lines_record = zeroContext(all_input_lines)
    else:
        all_input_lines, lines_record = nonZeroContext(all_input_lines, context)

    if mode == "train":
        # Making single word per line version of gold file
        new_output_lines = []
        for line in all_output_lines:
            line = line.strip().split()
            for word in line:
                new_output_lines.append(word)
        return all_input_lines, new_output_lines, lines_record

    return all_input_lines, lines_record