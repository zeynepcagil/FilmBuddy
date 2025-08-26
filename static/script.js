const form = document.getElementById('ask-form');
const queryInput = document.getElementById('query');
const chatHistory = document.getElementById('chat-history');

// Yeni mesajı sohbet geçmişine ekleyen fonksiyon
function addMessageToHistory(message, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${sender}-message`);
    const messageParagraph = document.createElement('p');
    messageParagraph.innerText = message;
    messageDiv.appendChild(messageParagraph);
    chatHistory.appendChild(messageDiv);

    // Sohbet geçmişini en aşağı kaydır
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return; // Boş sorguları engelle

    // Kullanıcının mesajını ekle
    addMessageToHistory(query, 'user');
    queryInput.value = ''; // Giriş alanını temizle

    // Asistanın yanıtını eklemek için geçici mesaj
    addMessageToHistory("Yanıt aranıyor...", 'assistant');

    // Geçici mesajı bulup düzenlemek için bir referans alalım
    const tempMessage = chatHistory.lastChild.querySelector('p');

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query }),
        });

        const data = await response.json();
        // Asistanın gerçek yanıtıyla geçici mesajı değiştir
        tempMessage.innerText = data.response;
    } catch (error) {
        tempMessage.innerText = 'Bir hata oluştu: ' + error.message;
    }
});