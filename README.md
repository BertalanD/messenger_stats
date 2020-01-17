# Messenger Statistics
Are you curious about your time on Facebook Messenger? Do you like noticing patterns in things all the time? Do you like being paranoid about how much Facebook knows about you? Discover these with statistics like: yearly, monthly, daily and hourly usage; most frequent words; most common reactions; number of messages by sender and the number and length of calls (buggy, does not work yet). 

## Usage
### Downloading your Messenger data
1. Open your [Facebook settings](https://www.facebook.com/settings?tab=your_facebook_information), navigate to *Your Facebook Information*, and click on *Download Your Information*
2. Uncheck every category except *Messages*. The *Friends* category gives you additional information, but is not strictly required.
3. Set *Format* to **JSON**. You should set *Media Quality* to **Low** to increase download speed.
4. Once it is available, download the ZIP archive.
### Install Python and dependencies
1. From your distribution's package manager or from the [official website](https://www.python.org/downloads/) download Python programming language. Version 3.7 or higher is required.
2. During installation, make sure *pip* gets installed as well.
3. Install dependencies with *pip*.
    ```
    pip install -r requirements.txt
    ```
### Generate HTML report
Run the `report.py` script. Answer the prompts to get the data you need.

Not implemented yet.

### Interactive analysis
Open a console in the directory where this file is located, then type 
```
jupyter notebook
``` 
From the file tree, open [`Messenger Statistics.ipynb`](./Statistics.ipynb). Follow the instructions therein.
