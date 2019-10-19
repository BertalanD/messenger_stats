import zipfile

class MessengerStats:
    """
    
    """

    def __init__(self, chats, participants, messages, participation, reactions):
        self.chats = chats
        self.participants = participants
        self.messages = messages
        self.participation = participation
        self.reactions = reactions

    @classmethod
    def from_zip(cls, zip_input: zipfile.ZipFile):
        print(cls)