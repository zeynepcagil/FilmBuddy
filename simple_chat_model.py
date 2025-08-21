import random


class SimpleChatModel:
    def __init__(self):
        # Temel ifadeler
        self.greetings = ["selam", "merhaba", "hi", "hey", "günaydın", "iyi günler"]
        self.how_are_you = ["nasılsın", "iyi misin", "keyfin nasıl", "ne haber", "ne var ne yok"]
        self.user_feelings = {
            "iyiyim": ["İyi olmana sevindim!", "Harika, mutlu olmana sevindim!", "Ne güzel, iyi olmana sevindim!"],
            "kötüyüm": ["Üzgünüm, umarım daha iyi hissedersin.", "Canın sıkkın gibi görünüyor, umarım toparlarsın.",
                        "Bunu duyduğuma üzüldüm."],
            "fena değil": ["Anladım, umarım günün daha iyi geçer.", "Hmm, umarım keyfin yerine gelir.",
                           "Tamam, yine de iyi olmana sevindim!"],
            "berbat": ["Üzgünüm, canın mı sıkıldı?", "Bunu duyduğuma üzüldüm, umarım düzelir.",
                       "Vay, kötüymüş, umarım günün toparlanır."],
            "yorgunum": ["Dinlenmek iyi gelir, umarım biraz rahatlayabilirsin.", "Anladım, dinlenmeye vakit ayır!",
                         "Umarım enerjin kısa sürede yerine gelir."],
            "harikayım": ["Harika! Neşen daim olsun!", "Bunu duymak güzel, keyfini çıkar!",
                          "Süper! Mutlu olmana sevindim."]
        }

        self.default_responses = ["Hmm, bunu tam anlayamadım ama selamlar!", "Emin değilim ama merhaba!"]

        # Durum takibi
        self.awaiting_how_are_you_response = False
        self.chat_history = []

    def respond(self, message: str) -> str:
        msg_lower = message.lower()
        self.chat_history.append(message)

        # Selamlaşma
        if any(greet in msg_lower for greet in self.greetings):
            self.awaiting_how_are_you_response = True
            return random.choice(["Merhaba!", "Selam!", "Hey! Nasılsın?"])

        # Nasılsın sorusuna cevap bekleniyor mu
        if self.awaiting_how_are_you_response:
            self.awaiting_how_are_you_response = False
            for feeling, responses in self.user_feelings.items():
                if feeling in msg_lower:
                    return random.choice(responses)
            return "Anladım, teşekkürler! Senin için her şey yolunda mı?"

        # Kullanıcı nasılsın soruyor mu
        if any(hay in msg_lower for hay in self.how_are_you):
            self.awaiting_how_are_you_response = True
            return random.choice(["İyiyim, ya sen?", "Gayet iyiyim, sen nasılsın?", "Harikayım, teşekkürler! Sen?"])

        # Diğer tanımadık mesajlar
        return random.choice(self.default_responses)




