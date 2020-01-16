import copy
import datetime
import json
import re
import unicodedata
import zipfile
import pytz
import tzlocal
import calendar
from typing import List, Optional, Tuple, Union
from collections import Counter
from string import punctuation

import numpy as np
import pandas as pd


class MessengerStats:
    """
    Facebook Messenger messages, contacts and reactions with functions to
    generate statistics and to filter and display them.


    Methods
    -------
    from_zip(zip_input: zipfile.ZipFile)
        Parses `zip_input` and constructs the MessengerStats object.
    TODO document others here
    FIXME: Crashes with empty dataframes
    """

    __slots__ = ["participants", "chats", "participation",
                 "messages", "reactions", "calls", "account_id", "timezone"]

    def __init__(self,  participants: pd.Series, chats: pd.DataFrame, participation: pd.DataFrame, messages: pd.DataFrame, reactions: pd.DataFrame, calls: pd.DataFrame, timezone=tzlocal.get_localzone(), merge=True):
        """
        Instantiate a new MessengerStats object.

        This function is not intended to be called by outside code!
        **Use the `from_zip()` method to load data.**
        """
        self.participants = participants
        self.chats = chats
        self.participation = participation
        self.messages = messages
        self.reactions = reactions
        self.calls = calls
        # Determine id (=index in self.participants) of the account whose data is being parsed
        # 1. The user present in most Regular (only 2 participants) chats
        regular_chats = chats[chats["thread_type"] == "Regular"]
        regular_participation = participation[[
            p in regular_chats.index for p in participation["chat_id"]]]
        in_most = participation["participant_id"].mode()
        if len(in_most) > 1:
            # 2: "Just You" conversation
            # TODO
            possible_ids = [0, 1, 2]
            if len(possible_ids) == 1:
                account_id = possible_ids[0]
            else:
                # 3. Person with most messages:
                most_messages = messages["sender_id"].mode()
                if len(most_messages) == 1:
                    account_id = most_messages[0]
                else:
                    raise Exception(
                        "Could not determine id of account. We're sorry, but execution can't continue.")
        else:
            account_id = in_most.iloc[0]
        self.account_id = account_id
        self.timezone = timezone
        if merge:
            new = self.merge_chats_with_same_name().merge_people_with_same_name()
            # self = new Does not work for some reason HACK
            self.participants = new.participants
            self.chats = new.chats
            self.participation = new.participation
            self.messages = new.messages
            self.reactions = new.reactions
            self.calls = new.calls
            self.account_id = new.account_id

    @classmethod
    def from_zip(cls, zip_input: zipfile.ZipFile, **kwargs):
        """
        Load Facebook Messenger data from the downloaded Zip archive.

        Parameters
        ----------
        zip_input: zipfile.ZipFile
            Zip archive containing Facebook Messenger data.
            Folder hierarchy should be 
            /messages/inbox[,archived_threads,...]/.*/message_1.json[,...]
        """
        participants_list: List[str] = []
        chats_list: List[Tuple[str, bool, str, str]] = []
        participation_list: List[Tuple[int, int]] = []
        messages_list: List[Tuple[int, int, pd.Timestamp, str, str]] = []
        reactions_list: List[Tuple[int, int, str]] = []
        calls_list: List[Tuple[int, int, pd.Timestamp, int, bool]] = []

        message_file_regex = re.compile(r"^messages\/.*message_\d+\.json$")
        chat_paths = filter(lambda p:  re.match(
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
                if msg_type == "Call":
                    duration = message["call_duration"]
                    missed = message.get("missed", False)
                    calls_list.append((chat_id, sender_id, timestamp_ms, duration, missed))
                content = message.get("content", None)
                content = _fix_encoding(
                    content) if not content == None else None
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
        calls = pd.DataFrame(
            data=calls_list,
            columns = {
                "chat_id": int, "caller_id": int, "timestamp": pd.Timestamp,
                "duration": int, "missed": bool
            }
        )
        return cls(participants=participants, chats=chats, participation=participation, messages=messages, reactions=reactions, calls=calls, **kwargs)

    def get_largest_chats(self) -> pd.DataFrame:
        """
        Get chats sorted by message count in decreasing order.

        Returns
        -------
        pandas.DataFrame
            Columns: 
                - `title`: name of the chat
                - `message_count`: number of messages
        """
        messages_by_chat = self.messages.groupby(
            ["chat_id"]).size().sort_values(ascending=False)
        largest_chat_titles = [self.chats.loc[x, "title"]
                               for x in messages_by_chat.index]
        largest_chats = pd.DataFrame(
            {"title": largest_chat_titles, "message_count": messages_by_chat})
        return largest_chats

    def get_message_count_by_sender(self) -> pd.DataFrame:
        """
        Get people sorted by number of messages sent in decreasing order.

        Returns
        -------
        pandas.DataFrame
            Columns:
                - `sender`: name of the sender
                - `message_count`: number of messages
        """
        messages_by_sender = self.messages.groupby(
            ["sender_id"]).size().sort_values(ascending=False)
        sender_names = [self.participants[x] for x in messages_by_sender.index]
        messages_by_sender = pd.DataFrame(
            {"sender": sender_names, "message_count": messages_by_sender})
        return messages_by_sender

    def get_messages_by_day_by_chat(self) -> pd.Series:
        """

        Returns
        -------
        pandas.DataFrame
            Index:
                - `title`
                - `date`
        """
        messages_with_day = copy.deepcopy(self.index_messages_by_time())
        messages_by_day_and_chat = messages_with_day.groupby(
            ["chat_id", "date"]).size()
        new_index = pd.MultiIndex.from_tuples(
            [(self.chats.loc[x[0], "title"], x[1])for x in messages_by_day_and_chat.index], names=["title", "date"])
        messages_by_day_and_chat.index = new_index
        return messages_by_day_and_chat

    def get_message_count_by_day_and_chat(self) -> pd.DataFrame:
        return self.get_messages_by_day_by_chat()

    def get_messages_by_hour_by_chat(self) -> pd.Series:
        """

        Returns
        -------
        pandas.Series
            Index:
                - `title`
                - `hour`
        """
        messages_with_hour = copy.deepcopy(self.index_messages_by_time())
        messages_with_hour = messages_with_hour.sort_values("hour")
        messages_by_hour_and_chat = messages_with_hour.groupby(
            ["chat_id", "hour"]).size()
        new_index = pd.MultiIndex.from_tuples(
            [(self.chats.loc[x[0], "title"], x[1]) for x in messages_by_hour_and_chat.index], names=["title", "hour"])
        messages_by_hour_and_chat.index = new_index
        return messages_by_hour_and_chat

    def get_message_count_by_hour_and_chat(self) -> pd.DataFrame:
        return self.get_messages_by_hour_by_chat()

    def get_messages_by_chat_hourly(self) -> pd.DataFrame:
        messages_with_hour = copy.deepcopy(self.index_messages_by_time())
        messages_by_chat_hourly = messages_with_hour.groupby(["chat_id", "date", "hour"]).size()
        new_index = pd.MultiIndex.from_tuples(
            [
                (
                    self.chats.loc[x[0], "title"],
                    pd.Timestamp(year=x[1].year, month=x[1].month, day=x[1].day, hour=x[2], minute=0, tzinfo=self.timezone)
                )
                for x in messages_by_chat_hourly.index
            ],
            names=["title","hourly"]
        )
        messages_by_chat_hourly.index = new_index
        return messages_by_chat_hourly

    def get_messages_by_weekday_by_chat(self) -> pd.DataFrame:
        """

        Returns
        -------
        pandas.Series
            Index:
                - `title`
                - `weekday`
        """
        messages_with_weekday = copy.deepcopy(
            self.index_messages_by_time()).sort_values("day_of_week_num")
        messages_by_weekday_and_chat = messages_with_weekday.groupby(
            ["chat_id", "day_of_week_num"]).size()
        new_index = pd.MultiIndex.from_tuples([(self.chats.loc[x[0], "title"], calendar.day_abbr[x[1]])
                                               for x in messages_by_weekday_and_chat.index], names=["title", "weekday"])
        messages_by_weekday_and_chat.index = new_index
        return messages_by_weekday_and_chat

    def get_message_count_by_weekday_and_chat(self) -> pd.DataFrame:
        return self.get_messages_by_weekday_by_chat()

    def get_messages_by_hour_of_week_by_chat(self) -> pd.Series:
        """

        Returns
        -------
        pandas.Series
            Index:
                - `title`
                - `day`: abbreviated name of day (**Need to fix day order of days when dispalying!**)
                - `hour`
        """
        messages_with_time = copy.deepcopy(self.index_messages_by_time()).sort_values([
            "day_of_week_num", "hour"])
        grouped = messages_with_time.groupby(
            ["chat_id", "day_of_week_num", 'hour']).size()
        new_index = pd.MultiIndex.from_tuples(
            [(self.chats.loc[x[0], "title"], calendar.day_abbr[x[1]], x[2]) for x in grouped.index], names=["title", "day", "hour"])
        grouped.index = new_index
        return grouped

    def get_time_range(self) -> Tuple[datetime.datetime, datetime.datetime]:
        """
        Returns
        -------
        tuple (datetime.datetime, datetime.datetime)
            - tuple[0]: time when the first message was sent
            - tuple[1]: time when the last message was sent
        """
        return (self.messages["timestamp"].min(), self.messages["timestamp"].max())

    def get_chats_by_participant(self) -> pd.Series:
        """
        Get the number of chats each person participates in.

        Returns
        -------
        pandas.Series
            Index:
                - `participant` 
        """
        # Quickly remove self
        participation = self.participation[[
            i != self.account_id for i in self.participation["participant_id"]]]
        chats_by_participant = participation.groupby(
            ["participant_id"]).size().sort_values(ascending=False)
        participant_names = [self.participants.loc[x]
                             for x in chats_by_participant.index]
        chats_by_participant = pd.Series(
            index=pd.Index(data=participant_names, names="participant"), data=chats_by_participant.values, dtype=int)
        return chats_by_participant

    def get_reaction_counts(self) -> pd.Series:
        """
        Get the total number of reactions for each type.

        Returns
        -------
        pd.Series
            Index:
                - `reaction`
        """
        return self.reactions.groupby(["reaction"]).size()

    def get_reactions_by_day(self) -> pd.Series:
        """
        Get the total number of reactions by day for each type.

        Returns
        -------
        pandas.Series
            Index:
                - `reaction`
                - `date`
        """
        reactions_extra = copy.deepcopy(self.reactions)
        local = self.timezone
        reactions_extra.loc[:, "timestamp_local"] = [pytz.utc.localize(
            self.messages.at[r, "timestamp"]).astimezone(local) for r in reactions_extra["message_id"]]
        reactions_extra.loc[:, "date"] = [t.date()
                                          for t in reactions_extra["timestamp_local"]]
        reactions_extra = reactions_extra.groupby(["reaction", "date"]).size()
        return reactions_extra

    def get_most_frequent_words(self):
        #words_to_count = map(lambda m: map(lambda w: w.lstrip(punctuation).lower(),msplit(" ")),self.messages["content"][[not pd.isna(c) for c in self.messages["content"]]])

        # FIXME: possibly removes more than just emoji
        emoji_pattern = re.compile('[\U00010000-\U0010ffff\u2700-\u27bf\u2665]', flags=re.UNICODE)
        words_to_count = [str(emoji_pattern.sub(r"",word)).lower().rstrip(punctuation).lstrip(punctuation) for message in self.messages["content"]
                          if message != None for word in re.split(r'[ \n\r\t\xa0]+', message)]
        
        counter = Counter(words_to_count)
        most_frequent_words = pd.DataFrame(list(counter.items()), columns=[
                                           "word", "count"]).sort_values("count", ascending=False)
        return most_frequent_words

    def get_top_chats_per_day(self) -> pd.Series:
        date_indexed = copy.copy(self.index_messages_by_time())
        messages_per_chat_per_day = date_indexed.groupby(["date", "chat_id"]).size().groupby("date").rank(ascending=False, method="max")
        new_index = pd.MultiIndex.from_tuples([(i[0], self.chats.loc[i[1], "title"]) for i in messages_per_chat_per_day.index], names=["date", "title"])
        messages_per_chat_per_day.index = new_index
        return messages_per_chat_per_day

    def get_calls_by_chat(self) -> pd.DataFrame:
        calls_by_chat = self.calls.groupby(
            ["chat_id"]).size().sort_values(ascending=False)
        largest_chat_titles = [self.chats.loc[x, "title"]
                               for x in calls_by_chat.index]
        largest_chats = pd.DataFrame(
            {"title": largest_chat_titles, "call_count": calls_by_chat})
        return largest_chats

    def get_call_duration_by_chat(self):
        duration_by_chat = self.calls.groupby(["chat_id"]).sum()["duration"].sort_values(ascending=False)
        chat_titles = [self.chats.loc[x, "title"]
                        for x in duration_by_chat.index]
        chats_by_duration = pd.DataFrame(
            {"title": chat_titles, "duration": duration_by_chat}
        )
        return chats_by_duration

    def filter_chat_types(self, chat_type: Union[List[str], str], negate: bool = False):
        """
        Filter chats based on their types.

        Parameters
        ----------
        chat_type: list of str or str
            Type(s) of chats to keep.
        negate: bool, default=False
            Invert operation. Keep chats whose type does not match.

        Returns
        -------
        MessengerStats
            Copy of `self` with chats, messages and reactions filtered.
        """

        new = copy.copy(self)
        # Changes in shallow copies DataFrames reflect onto the original,
        # So we have to make deep copies of values we modify

        chat_type = chat_type if isinstance(chat_type, list) else [chat_type]

        # Remove chats of other type
        if not new.chats.empty:
            new.chats = copy.deepcopy(self.chats)
            new.chats = new.chats[[
                (w in chat_type) ^ negate for w in new.chats["thread_type"]]]
        # Remove participations belonging to removed chats
        if not new.participation.empty:
            new.participation = copy.deepcopy(self.participation)
            new.participation = new.participation[[
                x in new.chats.index for x in new.participation["chat_id"]]]
        # Remove messages belonging to removed chats
        if not new.messages.empty:
            new.messages = copy.deepcopy(self.messages)
            new.messages = new.messages[[
                y in new.chats.index for y in new.messages["chat_id"]]]
        # Remove reactions belonging to removed messages
        if not new.reactions.empty:
            new.reactions = copy.deepcopy(self.reactions)
            new.reactions = new.reactions[[
                z in new.messages.index for z in new.reactions["message_id"]]]
        return new

    def filter_date(self, datetime_range: Union[List[Tuple[datetime.datetime, datetime.datetime]], Tuple[datetime.datetime, datetime.datetime]], negate: bool = False):
        """
        Filter chats based on the date they were sent.

        Parameters
        ----------
        datetime_range: tuple (datetime.datetime, datetime.datetime) or list of such tuples
            Discard messages and reactions sent outside of these interval(s).

            **Note:** You might want to add timezone information and set 
            data timezone using `set_timezone()`.

        negate: bool, default=False
            Invert operation. Discard messages within `datetime_range`

        Returns
        -------
        MessengerStats
            Copy of `self` with messages, calls and reactions filtered.
        """

        new = copy.copy(self)
        # Changes in shallow copies DataFrames reflect onto the original,
        # So we have to make deep copies of values we modify.

        datetime_range = datetime_range if isinstance(
            datetime_range, list) else [datetime_range]

        # Fix offset-naive input
        datetime_range = [(_make_offset_aware(t[0], self.timezone), _make_offset_aware(
            t[1], self.timezone)) for t in datetime_range]

        # Remove messages outside specified time ranges
        if not new.messages.empty:
            new.messages = copy.deepcopy(self.messages)
            new.messages = new.messages[[((True in [(r[0] <= pytz.utc.localize(t).astimezone(self.timezone) and pytz.utc.localize(t).astimezone(self.timezone) <= r[1])
                                        for r in datetime_range]) ^ negate) for t in new.messages["timestamp"]]]
        if not new.calls.empty:
            new.calls = copy.deepcopy(self.calls)
            new.calls = new.calls[[((True in [(r[0] <= pytz.utc.localize(t).astimezone(self.timezone) and pytz.utc.localize(t).astimezone(self.timezone) <= r[1])
                                    for r in datetime_range]) ^ negate) for t in new.calls["timestamp"]]]
        # Remove reactions belonging to removed messages
        if not new.reactions.empty:
            new.reactions = copy.deepcopy(self.reactions)
            new.reactions = new.reactions[[
                i in new.messages.index for i in new.reactions["message_id"]]]
        return new

    def filter_time(self, time_range: Union[Tuple[datetime.time, datetime.time], List[Tuple[datetime.time, datetime.time]]], negate: bool = False):
        """
        Filter chats based on the time (hour, minutes) they were sent.

        Parameters
        ----------
        time_range: tuple (datetime.time, datetime.time) or list of such tuples
            Discard messages and reactions sent outside of these interval(s).

            **Note:** You might want to add timezone information and set 
            data timezone using `set_timezone()`.

        negate: bool, default=False
            Invert operation. Discard messages within `time_range`

        Returns
        -------
        MessengerStats
            Copy of `self` with messages and reactions filtered.
        """

        new = copy.copy(self)
        # Changes in shallow copies DataFrames reflect onto the original,
        # So we have to make deep copies of values we modify.

        time_range = time_range if isinstance(
            time_range, list) else [time_range]

        # Fix offset-naive input
        time_range = [(_make_offset_aware(t[0], self.timezone), _make_offset_aware(
            t[1], self.timezone)) for t in time_range]
        if not new.messages.empty:
            new.messages = copy.deepcopy(self.messages)
            # Remove messages outside specified time ranges
            new.messages = new.messages[[((True in [(r[0] <= pytz.utc.localize(t).astimezone(self.timezone).timetz() and pytz.utc.localize(t).astimezone(self.timezone).timetz() <= r[1])
                                            for r in time_range]) ^ negate) for t in new.messages["timestamp"]]]
        if not new.calls.empty:
            new.calls = copy.deepcopy(self.calls)
            new.calls = new.calls[[((True in [(r[0] <= pytz.utc.localize(t).astimezone(self.timezone).timetz() and pytz.utc.localize(t).astimezone(self.timezone).timetz() <= r[1])
                                    for r in time_range]) ^ negate) for t in new.calls["timestamp"]]]
        # Remove reactions belonging to removed messages
        if not new.reactions.empty:
            new.reactions = copy.deepcopy(self.reactions)
            new.reactions = new.reactions[[
                i in new.messages.index for i in new.reactions["message_id"]]]
        return new

    def filter_chats(self, chat_name: Union[List[str], str], negate: bool = False):
        """
        Filter chats based on their names.

        Parameters
        ----------
        chat_name: list of str or str
            Names of chats to keep.

        negate: bool, default=False
            Invert operation. Discard messages whose name matches.

        Returns
        -------
        MessengerStats
            Copy of `self` with chats, messages, calls and reactions filtered.
        """

        new = copy.copy(self)
        # Changes in shallow copies DataFrames reflect onto the original,
        # So we have to make deep copies of values we modify.

        chat_names = chat_name if isinstance(chat_name, list) else [chat_name]
        if not new.chats.empty:
            new.chats = copy.deepcopy(self.chats)
            new.chats = new.chats[[((t in chat_names) ^ negate)
                                for t in new.chats["title"]]]
        # Remove participations belonging to removed chats
        if not new.participation.empty:
            new.participation = copy.deepcopy(self.participation)
            new.participation = new.participation[[
                (x in new.chats.index) for x in new.participation["chat_id"]]]
        # Remove messages belonging to removed chats
        if not new.messages.empty:
            new.messages = copy.deepcopy(self.messages)
            new.messages = new.messages[[(i in new.chats.index)
                                        for i in new.messages["chat_id"]]]
        if not new.calls.empty:
            new.calls = copy.deepcopy(self.calls)
            new.calls = new.calls[[(i in new.chats.index)
                                    for i in new.calls["chat_id"]]]
        # Remove reactions belonging to removed messages
        if not new.reactions.empty:
            new.reactions = copy.deepcopy(self.reactions)
            new.reactions = new.reactions[[
                (i in new.messages.index) for i in new.reactions["message_id"]]]
        return new

    def filter_people(self, name: Union[List[str], str], negate: bool = False):
        """
        Filter messages and reactions by sender and calls by caller.

        Parameters
        ----------
        name: list of str or str
            Sender(s) whose messages and reactions to keep.

        negate: bool, default=False
            Invert operation. Discard messages and reactions by `sender_name`.

        Returns
        -------
        MessengerStats
            Copy of `self` with messages and reactions filtered.
        """

        new = copy.copy(self)
        # Changes in shallow copies DataFrames reflect onto the original,
        # So we have to make deep copies of values we modify.
        new.participants = copy.deepcopy(self.participants)
        
        names = name if isinstance(name, list) == str else [name]
        ids = [self.name_to_id(n) for n in names]
        new.participants = new.participants[[
            (p in ids) ^ negate for p in new.participants.index]]
        # Remove participations belonging to removed participants
        if not new.participation.empty:
            new.participation = copy.deepcopy(self.participation)
            new.participation = new.participation[[
                (n in ids) ^ negate for n in new.participation["participant_id"]]]
        if not new.messages.empty:
            new.messages = copy.deepcopy(self.messages)
            # Remove messages not belonging to name
            new.messages = new.messages[[(n in ids) ^ negate
                                        for n in new.messages["sender_id"]]]
        if not new.calls.empty:
            new.calls = copy.deepcopy(self.calls)
            new.calls = new.calls[[(n in ids) ^ negate
                                    for n in new.calls["caller_id"]]]
        # Remove reactions belonging to removed messages
        if not new.reactions.empty:
            new.reactions = copy.deepcopy(self.reactions)
            new.reactions = new.reactions[[
                (i in new.messages.index) for i in new.reactions["message_id"]]]
        if not new.chats.empty:
            new.chats = copy.deepcopy(self.chats)
            new.chats = new.chats[[((c in new.participation["chat_id"].values) or (
                c in new.messages["chat_id"].values)) for c in new.chats.index]]
        return new

    def filter_reactors(self, name: Union[List[str], str], negate: bool = False):
        """
        Filter reactions by reactor.

        Parameters
        ----------
        name: list of str or str
            Sender(s) whose reactions to keep.

        negate: bool, default=False
            Invert operation. Discard reactions by `sender_name`.

        Returns
        -------
        MessengerStats
            Copy of `self` with reactions filtered.
        """

        new = copy.copy(self)
        # Changes in shallow copies DataFrames reflect onto the original,
        # So we have to make deep copies of values we modify.

        reactors = name if isinstance(name, list) else [name]
        reactor_ids = [(self.name_to_id(n)) for n in reactors]
        if not new.reactions.empty:
            new.reactions = copy.deepcopy(self.reactions)
            # Crashes if it doesn't find reactor_id column
            new.reactions = new.reactions[[
                (i in reactor_ids) ^ negate for i in new.reactions["reactor_id"]]]
        return new

    def filter_message_senders(self, sender: Union[List[str], str], negate: bool = False):
        """
        Filter messages by sender.

        Parameters
        ----------
        sender: list of str or str
            Sender(s) whose messages to keep.

        negate: bool, default=False
            Invert operation. Discard messages by `sender_name`.

        Returns
        -------
        MessengerStats
            Copy of `self` with messages filtered.
        """

        new = copy.copy(self)
        # Changes in shallow copies DataFrames reflect onto the original,
        # So we have to make deep copies of values we modify.

        senders = sender if isinstance(sender, list) else [sender]
        sender_ids = [(self.name_to_id(n)) for n in senders]
        if not new.messages.empty:
            new.messages = copy.deepcopy(self.messages)
            new.messages = new.messages[[((i in sender_ids) ^ negate)
                                        for i in new.messages["sender_id"]]]
        # Remove orphaned reactions
        if not new.reactions.empty:
            new.reactions = copy.deepcopy(self.reactions)
            new.reactions = new.reactions[[
                (i in new.messages.index) for i in new.reactions["message_id"]]]
        return new

    def filter_callers(self, caller: Union[List[str], str], negate: bool = False):
        new = copy.copy(self)

        callers = caller if isinstance(caller, list) else [caller]
        caller_ids = [(self.name_to_id(n)) for n in callers]
        if not new.calls.empty:
            new.calls = copy.copy(new.calls)
            new.calls = new.calls[[(c in caller_ids) for c in new.calls["caller_id"]]]
        return new

    def filter_message_types(self, type: Union[List[str], str], negate: bool = False):
        """
        Filter messages by type.

        Parameters
        ----------
        type: list of str or str
            Type(s) of messages to keep.

        negate: bool, default=False
            Invert operation. Messages of type `type`.

        Returns
        -------
        MessengerStats
            Copy of `self` with messages filtered.
        """
        new = copy.copy(self)
        # Changes in shallow copies DataFrames reflect onto the original,
        # So we have to make deep copies of values we modify.

        types = type if isinstance(type, list) else [type]
        if not new.messages.empty:
            new.messages = copy.deepcopy(self.messages)
            new.messages = new.messages[[((m in types) ^ negate)
                                        for m in new.messages["type"]]]
        if not new.reactions.empty:
            new.reactions = copy.deepcopy(new.reactions)
            new.reactions = new.reactions[[
                (i in new.messages.index) for i in new.reactions["message_id"]]]
        return new

    def filter_own_messages(self, negate: bool = False):
        """ 
        Only keep own messages. 

        Parameters
        ----------
        negate: bool, default=False
            Invert operation. Returns all messages not sent by account.

        Returns
        -------
        MessengerStats
            Copy of `self` with messages filtered
        """
        return self.filter_message_senders(self.account_name(), negate=negate)

    def filter_own_calls(self, negate: bool = False):
        return self.filter_callers(self.account_name(), negate=negate)

    def name_to_id(self, name: str) -> Optional[str]:
        if name in self.participants.values:
            return np.where(self.participants == name)[0]
        else:
            return None

    def anonymize(self, group_prefix: str = "G", regular_prefix: str = "C", self_name: str = "Me"):
        """
        Remove chat titles and names that identify participants.
        Useful, for example, for the purpose of demonstration.

        Parameters
        ----------
        group_prefix: str, default="G"
            Prefix group chat names with this string.
        regular_prefix: str, default="C"
            Prefix regular conversations (2 participants) with this strings.
        self_name: str, default="Me"
            Set the name of the person whose data we're parsing.

        Returns
        -------
        MessengerStats
            Copy of `self` with chat titles and names converted to id.
        """
        new = copy.copy(self)
        # Changes in shallow copies DataFrames reflect onto the original,
        # So we have to make deep copies of values we modify.
        new.chats = copy.deepcopy(self.chats)
        new.participants = copy.deepcopy(self.participants)
        # Replace two-person chat names with other party
        new.chats["title"] = [
            regular_prefix+str(new.get_other_participant_ids(int(i))[0])
            if new.chats.loc[int(i), "thread_type"] == "Regular"
            else group_prefix+str(i)
            for (i, row) in new.chats.iterrows()
        ]
        # Replace names with id, except for self
        new.participants = pd.Series(
            [self_name if i == new.account_id else str(i) for i in new.participants.index])
        return new

    def account_name(self):
        """
        Name of the person whose data we are parsing.
        """
        return self.participants.at[self.account_id]

    def get_other_participant_ids(self, chat_id: int):
        """
        Get participants of chat `chat_id`, not including `account_name()`.
        """
        participants = self.participation[self.participation["chat_id"]
                                          == chat_id]["participant_id"]
        not_self = [None if p ==
                    self.account_id else p for p in participants]
        return not_self

    def index_messages_by_time(self, timezone: Optional[datetime.tzinfo] = None) -> pd.DataFrame:
        """
        Add detailed date/time data to messages.

        Parameters
        ----------
        timezone: datetime.tzinfo, default=self.timezone (local time if not explicitly set)
            Timezone that timestamps should be converted to. We have no
            way of getting the timezone that the messages were sent in.
            By default, this is the local timezone.

        Returns
        -------
        pd.DataFrame
            Messages with columns `timestamp_local`, `year`, `month`, `month_num`,
            `day_of_week`, `day_of_week_num` and `hour` added and sorted by time.
        """
        timezone = self.timezone if timezone == None else timezone
        time_indexed = copy.deepcopy(self.messages)
        time_indexed["message_id"] = time_indexed.index.values
        time_indexed["timestamp_local"] = [pytz.utc.localize(
            t).astimezone(timezone) for t in time_indexed["timestamp"]]
        time_indexed["date"] = [t.date()
                                for t in time_indexed["timestamp_local"]]
        time_indexed = time_indexed.set_index("timestamp_local")
        time_indexed["year"] = time_indexed.index.year
        time_indexed["month"] = time_indexed.index.strftime("%b")
        time_indexed["month_num"] = time_indexed.index.month
        time_indexed["day_of_week"] = time_indexed.index.strftime("%a")
        time_indexed["day_of_week_num"] = time_indexed.index.weekday
        time_indexed["hour"] = time_indexed.index.hour
        time_indexed.sort_values("timestamp", inplace=True)
        return time_indexed

    def get_friendly_messages(self) -> pd.DataFrame:
        """
        Get messages in a human-readable format. 

        Returns
        -------
        pandas.DataFrame
            Messages with timestamp converted to local time, sender ids
            converted to names, message ids converted to titles.
        """
        time_indexed = self.index_messages_by_time()
        time_indexed["sender"] = [self.participants[n]
                                  for n in time_indexed["sender_id"]]
        time_indexed["chat"] = [self.chats.loc[n, "title"]
                                for n in time_indexed["chat_id"]]
        time_indexed.sort_values(["chat_id", "timestamp_local"], inplace=True)
        time_indexed.set_index("message_id", inplace=True)
        return time_indexed

    def get_most_frequent_words_by_day(self):
        # FIXME: UNIMPLEMENTED
        pass

    def set_timezone(self, timezone: datetime.tzinfo):
        """
        Parameters
        ----------
        timezone: datetime.tzinfo
            Timezone that will be used to offset timestamps

        Returns
        -------
        MessengerStats
            Copy of `self` with timezone set to `timezone`.
        """
        new = copy.copy(self)
        new.timezone = timezone
        return new

    def merge_chats_with_same_name(self):
        new = copy.copy(self)
        new.chats = copy.deepcopy(new.chats)
        new.participation = copy.deepcopy(new.participation)
        new.messages = copy.deepcopy(new.messages)
        new.calls = copy.deepcopy(new.calls)
        for index, chat in new.chats.iterrows():
            if chat["title"] in new.chats[new.chats.index < index]["title"].values:
                new.chats.drop(index, inplace=True)
            new_index = new.chats.title[new.chats.title == chat["title"]].index[0]
            # Replace every occurrence of chat ID with the new one.
            new.participation.replace({"chat_id": index}, value=new_index, inplace=True)
            new.messages.replace({"chat_id": index}, value=new_index, inplace=True)
            new.calls.replace({"chat_id": index}, value=new_index, inplace=True)
        new.participation.drop_duplicates(inplace=True)
        return new

    def merge_people_with_same_name(self):
        # FIXME: Will create chaos if there are multiple people named own_account
        new = copy.copy(self)
        new.participants = copy.deepcopy(new.participants)
        new.participation = copy.deepcopy(new.participation)
        new.messages = copy.deepcopy(new.messages)
        new.reactions = copy.deepcopy(new.reactions)
        new.calls = copy.deepcopy(new.calls)
        for index, name in new.participants.iteritems():
            new_index = new.participants[new.participants == name].index[0]
            new.participants.replace(index,new_index, inplace=True)
            new.participation.replace({"participant_id": index}, new_index, inplace=True)
            new.messages.replace({"sender_id": index}, new_index, inplace=True)
            new.reactions.replace({"reactor_id": index}, new_index, inplace=True)
            new.calls.replace({"caller_id": index}, new_index, inplace=True)
        return new

def _fix_encoding(messed_up_unicode_string: str) -> str:
    r"""
    Fix messed-up UTF-8 encoding.

    Returns
    -------
    str
        Characters intended to be represented by the erroneous string.

    Notes
    -----
    For some unexplainable reason, Facebook encodes Unicode characters
    by encoding a separate codepoint for each of the bytes of the
    characters's UTF-8 representation.
    Thus 'é' (U+00E9 Latin Small Letter E with Acute) becomes
    encoded in JSON as \u00c3\u00a9, that Python correctly decodes
    to "Ã©" (U+00C3 Latin Capital Letter A with Tilde, U+00A9 Copyright Sign).


    """
    return messed_up_unicode_string.encode("latin1").decode("utf-8")


def _make_offset_aware(timeobj: Union[datetime.time, datetime.datetime], tzinfo: datetime.tzinfo = pytz.utc) -> Union[datetime.time, datetime.datetime]:
    return timeobj if timeobj.tzinfo is not None and timeobj.tzinfo.utcoffset(timeobj) is not None else timeobj.replace(tzinfo=tzinfo)
