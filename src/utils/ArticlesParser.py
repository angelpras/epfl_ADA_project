import os
import re
from collections import Counter
def parse_wiki_article(text):
    """
    Parse a Wikipedia article text to extract title, subjects, and first meaningful paragraph.
    """
    # Split into lines, preserving empty lines
    lines = text.split('\n')
    
    # Initialize return values
    title = None
    subjects = []
    first_paragraph = None
    
    # Find the Wikipedia Selection line with subjects
    subjects_line_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("2007 Schools Wikipedia Selection. Related subjects:"):
            subjects_line = line.strip()

            # Might have many lines pertaining to related subjects. Concatenate them.
            while lines[i+1] != '':
                subjects_line = ' '.join([subjects_line, lines[i+1].strip()])
                del lines[i+1]

            subjects_text = subjects_line.split("Related subjects:")[-1].strip()
            subjects = [subj.strip() for subj in subjects_text.split(";")]
            subjects_line_idx = i
            break

    # Get title (should be the non-empty line before subjects line)
    for i in range(subjects_line_idx - 1, -1, -1):
        if lines[i].strip() and not lines[i].strip().startswith("#"):
            title = lines[i].strip()
            break
    
     # Find the next section start
    next_section_start = -1
    for i in range(subjects_line_idx + 1, len(lines)):
        line = lines[i]
        if line and not line.startswith("  "):  # Line starts at beginning
            if line != title and not line.startswith("2007 Schools Wikipedia Selection"):
                next_section_start = i
                break

    # If we didn't find a next section, set it to the end of the file
    if next_section_start == -1:
        next_section_start = len(lines)
    
    section_text = "\n".join(lines[subjects_line_idx+1:next_section_start])

    paragraphs = section_text.split('\n\n')  # Splitting paragraphs by double line breaks
    
    for i, paragraph in enumerate(paragraphs):
        paragraph_lines = paragraph.strip('\n').split('\n')
        paragraph_lines = [line.strip() for line in paragraph_lines]
        paragraph_lines = remove_duplicates_and_enlarge(paragraph_lines)
        
        # Paragraphs should start with an uppercase letter (punctuated or not) or a symbol,
        # and end with either of .!?) symbols
        if len(paragraph_lines) == 0 or \
           not re.match(r'^[a-zA-ZÀ-ÖØ-Þ0-9\W]', paragraph_lines[0]) or \
           not re.match(r'.*[.!?)\"\n]$', paragraph_lines[-1]):
            continue
            
        # Check if there is any line break that shouldn't ha ppen
        is_fake_paragraph = False

        for i in range(len(paragraph_lines) - 1):  # Exclude last line
            current_line = paragraph_lines[i] + ' ' + paragraph_lines[i + 1].split(' ')[0]

            # If there is a line break and the combined length of the lines is less than K, it's a fake paragraph
            if len(current_line) < 68:
                is_fake_paragraph = True
                # print(len(current_line))
                # print(current_line)
                break
            
        if not is_fake_paragraph:
            # If it's not a fake paragraph, set the first paragraph
            first_paragraph = ' '.join(line for line in paragraph_lines)
            break  # Stop after finding the first valid paragraph
    
    return title, subjects, first_paragraph

def process_articles_directory(directory_path):
    """Process all text files in a directory and extract article information."""
    results = []
    
    for filename in os.listdir(directory_path):
        if not filename.endswith('.txt'):
            continue
            
        filepath = os.path.join(directory_path, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            try:
                title, subjects, paragraph = parse_wiki_article(content)
            except:
                print(f"Article {filename} could not be parsed")
            if title and paragraph:
                results.append((title, subjects, paragraph))
            else:
                print(f"Article {filename} could not be parsed")
    
    return results

def remove_duplicates_and_enlarge(lines):
    # Count the occurrences of each line
    line_counts = Counter(lines)
    lines = [line for line in lines if line_counts[line] == 1 and line != 'Enlarge']
    # Only keep lines that appear exactly once
    return lines

# Example usage
if __name__ == "__main__":
    # Test with a sample that has section structure
    sample_text = """   #copyright

Finance

2007 Schools Wikipedia Selection. Related subjects: Economics

   Finance

   Finance Overview
   Capital
   Investment
   Cash flow
   Credit
   Debt
   Funding
   Hedging
   Interest
   Risk
   Yield
   Arbitrage

   Types of Finance
   Corporate finance
   Personal finance
   Public finance

   Asset Types
   Real Estate
   Securities
   Commodities
   Futures

   Financial Vehicles
   Collective Investment Schemes
   Trusts

   See Also
   Entrepreneur
   Financial market
   Insurance
   Economy
                   edit this box

   Finance studies and addresses the ways in which individuals, businesses,
   and organizations raise, allocate, and use monetary resources over
   time, taking into account the risks entailed in their projects.
   
   The term finance may thus incorporate any of the following:
     * The study of money and other assets
     * The management and control of those assets
     * Profiling and managing project risks
     * As a verb, "to finance" is to provide funds for business.

Examples of some basic financial concepts
"""

    title, subjects, para = parse_wiki_article(sample_text)
    print(f"Title: {title}")
    print(f"Subjects: {subjects}")
    print(f"First Paragraph: {para}")