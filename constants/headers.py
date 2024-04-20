REQUESTS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

HUGGINGCHAT_POST_HEADERS = {
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Content-Type": "application/json",
    "Origin": "https://huggingface.co",
    "Pragma": "no-cache",
    "Referer": "https://huggingface.co/chat/",
    "Sec-Ch-Ua": 'Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
}

HUGGINGCHAT_SETTINGS_POST_DATA = {
    "assistants": [],
    "customPrompts": {},
    "ethicsModalAccepted": True,
    "ethicsModalAcceptedAt": None,
    "hideEmojiOnSidebar": False,
    "recentlySaved": False,
    "searchEnabled": True,
    "shareConversationsWithModelAuthors": True,
}

OPENAI_GET_HEADERS = {
    # "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Content-Type": "application/json",
    # "Oai-Device-Id": self.uuid,
    "Oai-Language": "en-US",
    "Pragma": "no-cache",
    "Referer": "https://chat.openai.com/",
    "Sec-Ch-Ua": 'Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
}


OPENAI_POST_DATA = {
    "action": "next",
    # "conversation_id": "...",
    "conversation_mode": {"kind": "primary_assistant"},
    "force_nulligen": False,
    "force_paragen": False,
    "force_paragen_model_slug": "",
    "force_rate_limit": False,
    "history_and_training_disabled": False,
    # "messages": [...],
    "model": "text-davinci-002-render-sha",
    "parent_message_id": "",
    "suggestions": [],
    "timezone_offset_min": -480,
}
