import zipfile
import pandas as pd
import re
import json
import unicodedata
import datetime
import copy 
from typing import List, Tuple, Union


class MessengerStats:
    """Facebook Messenger messages and associated data.

    Methods
    -------
    from_zip(zip_input)
        Parses zip_input and constructs the MessengerStats object.
    """

    def __init__(self,  participants: pd.Series, chats: pd.DataFrame, participation: pd.DataFrame, messages: pd.DataFrame, reactions: pd.DataFrame):
        self.participants = participants
        self.chats = chats
        self.participation = participation
        self.messages = messages
        self.reactions = reactions

    @classmethod
    def from_zip(cls, zip_input: zipfile.ZipFile):
        participants_list: List[str] = []
        chats_list: List[Tuple[str, bool, str, str]] = []
        participation_list: List[Tuple[int, int]] = []
        messages_list: List[Tuple[int, int, pd.Timestamp, str, str]] = []
        reactions_list: List[Tuple[int, int, str]] = []

        message_file_regex = re.compile(r"^messages\/.*message_\d+\.json$")
        chat_paths = filter(lambda p: re.match(
            message_file_regex, p) != None, zip_input.namelist())

        for json_file in chat_paths:
            chat = json.load(zip_input.open(json_file))
            title = _fix_encoding(chat["title"])
            is_still_participant: bool = chat["is_still_participant"]
            thread_type = chat["thread_type"]
            thread_path = chat["thread_path"]
            chats_list.append(
                (title, is_still_participant, thread_type, thread_path))
            chat_id = len(chats_list) - 1

            participants_in_chat: List[Tuple[int, str]] = []

            def name_to_id(name: str) -> int:
                nonlocal participants_list
                nonlocal participants_in_chat
                for user_tuple in participants_in_chat:
                    if user_tuple[1] == name:
                        return user_tuple[0]
                try:
                    index = participants_list.index(name)
                    participants_in_chat.append((index, name))
                    return index
                except:
                    participants_list.append(name)
                    new_id = participants_list.index(name)
                    participants_in_chat.append((new_id, name))
                    return new_id
                return None

            for participant in chat["participants"]:
                participant_name = _fix_encoding(participant["name"])
                participant_id = name_to_id(participant_name)
                participation_list.append((chat_id, participant_id))

            for message in chat["messages"]:
                sender_name = _fix_encoding(message["sender_name"])
                sender_id = name_to_id(sender_name)
                timestamp_ms = pd.Timestamp(message["timestamp_ms"], unit="ms")
                msg_type = message["type"]
                content = message.get("content", None)
                content = content.encode("latin1").decode(
                    "utf-8") if not content == None else None
                messages_list.append(
                    (chat_id, sender_id, timestamp_ms, content, msg_type))
                msg_id = len(messages_list) - 1
                if "reactions" in message:
                    for reaction in message["reactions"]:
                        reaction_kind = _fix_encoding(reaction["reaction"])
                        reactor = _fix_encoding(reaction["actor"])
                        reactor_id = name_to_id(reactor)
                        reactions_list.append(
                            (msg_id, reactor_id, reaction_kind))
        participants = pd.Series(data=participants_list, dtype=str)
        chats = pd.DataFrame(
            data=chats_list,
            columns={"title": str, "is_still_participant": bool,
                     "thread_type": "category", "thread_path": str},
        )
        chats["thread_type"] = chats["thread_type"].astype("category")
        participation = pd.DataFrame(
            data=participation_list,
            columns={"chat_id": int, "participant_id": int},
        )
        messages = pd.DataFrame(
            data=messages_list,
            columns={"chat_id": int, "sender_id": int,
                     "timestamp": pd.Timestamp, "content": str, "type": "category"}
        )
        messages["type"] = messages["type"].astype("category")
        reactions = pd.DataFrame(
            data=reactions_list,
            columns={"message_id": int,
                     "reactor_id": int, "reaction": "category"}
        )
        reactions["reaction"] = reactions["reaction"].astype("category")
        return cls(participants, chats, participation, messages, reactions)

    def get_regular_chats(self) -> pd.DataFrame:
        filtered = self.chats
        is_regular = filtered["thread_type"] == "Regular"
        filtered = filtered[is_regular]
        return filtered

    def get_largest_chats(self, only_regular=False) -> pd.DataFrame:
        messages_by_chat = self.messages.groupby(
            ["chat_id"]).size().sort_values(ascending=False)
        largest_chat_titles = [self.chats.loc[x].at["title"]
                               for x in messages_by_chat.index.values]
        largest_chats = pd.DataFrame(
            {"title": largest_chat_titles, "message_count": messages_by_chat})
        return largest_chats

    def get_time_range(self) -> Tuple[datetime.datetime, datetime.datetime]:
        return (self.messages["timestamp"].min(), self.messages["timestamp"].max())

    def filter_chat_types(self, chat_type: Union[List[str],str]):
        new = copy.deepcopy(self)
        # Remove chats of other type
        if type(chat_type) == str:
            new.chats = new.chats[new.chats["thread_type"] == chat_type]
        else:
            new.chats = new.chats[[(w in chat_type) for w in new.chats["thread_type"]]]
        # Remove participations belonging to removed chats
        new.participation = new.participation[[(x in new.chats.index) for x in new.participation["chat_id"]]]
        # Remove messages belonging to removed chats
        new.messages = new.messages[[(y in new.chats.index) for y in new.messages["chat_id"]]]
        # Remove reactions belonging to removed messages
        new.reactions = new.reactions[[(z in new.messages.index) for z in new.reactions["message_id"]]]
        return new

    def filter_chat_types_not(self, chat_type: Union[List[str], str]):
        all = self.chats["thread_type"].cat.categories
        chat_type = [chat_type] if type(chat_type) == str else chat_type # type: ignore
        types_to_keep = [item for item in all if item not in chat_type]
        new = copy.deepcopy(self)
        new = new.filter_chat_types(types_to_keep)
        return new

    def filter_time(self, time_ranges: List[Tuple[datetime.datetime, datetime.datetime]]):
        new = copy.deepcopy(self)
        # Remove messages outside specified time ranges
        new.messages = new.messages[[True in [(r[0] <= t and t <= r[1]) for r in time_ranges] for t in new.messages["timestamp"]]]
        return new

def _fix_encoding(messed_up_unicode_string: str) -> str:
    """Fix messed-up UTF-8 encoding"""
    return unicodedata.normalize("NFKC", messed_up_unicode_string.encode("latin1").decode("utf-8"))
