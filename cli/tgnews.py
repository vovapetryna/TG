import sys
import os

# python tgnews languages <source_dir>
# python tgnews news <source_dir>
# python tgnews categories <source_dir>
# python tgnews threads <source_dir>

COMMAND_INDEX = 1
FOLDER_INDEX = 2

for i in range(len(sys.argv)):
    print("iteration : {}, arg: {}".format(i, sys.argv[i]))

def get_html_files(path):
    # https://mkyong.com/python/python-how-to-list-all-files-in-a-directory/
    result = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if '.html' in file:
                result.append(os.path.join(root, file))

    for f in result:
        print(f)

    return result

def language_handle(file_names_list):
    # get all samples and group them by languige en, ru

def news_handle(file_names_list):
    # get only news and print them

def categories_handle(file_names_list):
    # sort by categories

def threads_handle(file_names_list):
    # do thread stuff

if len(sys.argv) < FOLDER_INDEX:
    exit(0)

input_filenames = get_html_files(sys.argv[FOLDER_INDEX])

if sys.argv[COMMAND_INDEX] == "language":
    language_handle(input_filenames)

if sys.argv[COMMAND_INDEX] == "news":
    news_handle(input_filenames)

if sys.argv[COMMAND_INDEX] == "categories":
    categories_handle(input_filenames)

if sys.argv[COMMAND_INDEX] == "threads":
    threads_handle(input_filenames)