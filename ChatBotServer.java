import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;

/**
 * AI Chatbot (Enhanced) - CodeAlpha Task 3
 * ------------------------------------------
 * A Java-based chatbot with a web interface (HTML/CSS/JS served by an
 * embedded Java HTTP server). Combines several NLP + ML-style techniques:
 *
 *   1. Tokenization, punctuation stripping, stop-word removal
 *   2. Basic stemming (plural / common suffix normalization)
 *   3. Synonym normalization (maps related words to one canonical term)
 *   4. TF-IDF-weighted similarity scoring (rarer, more distinctive words
 *      count more than common ones - a lightweight ML-style ranking method)
 *   5. Session-based context memory, so short follow-up questions like
 *      "how much?" after asking about fees are understood in context
 *   6. "Did you mean...?" smart suggestions when no confident match is found
 *
 * No external libraries required - uses only the built-in JDK HttpServer.
 *
 * Run:
 *   javac ChatBotServer.java
 *   java ChatBotServer
 * Then open: http://localhost:8080
 */
public class ChatBotServer {

    private static final int PORT = 8080;
    private static final ChatEngine engine = new ChatEngine();

    public static void main(String[] args) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(PORT), 0);
        server.createContext("/", ChatBotServer::handleHome);
        server.createContext("/chat", ChatBotServer::handleChat);
        server.setExecutor(Executors.newFixedThreadPool(4));
        server.start();
        System.out.println("=====================================================");
        System.out.println(" AI Campus Assistant Chatbot Server started!");
        System.out.println(" Open your browser and go to: http://localhost:" + PORT);
        System.out.println(" Press CTRL+C to stop the server.");
        System.out.println("=====================================================");
    }

    // ---------------- HTTP Handlers ----------------

    private static void handleHome(HttpExchange exchange) throws IOException {
        byte[] bytes = HTML_PAGE.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", "text/html; charset=UTF-8");
        exchange.sendResponseHeaders(200, bytes.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static void handleChat(HttpExchange exchange) throws IOException {
        if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
            exchange.sendResponseHeaders(405, -1);
            return;
        }
        String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        String userMessage = JsonUtil.extractValue(body, "message");
        String sessionId = JsonUtil.extractValue(body, "session");
        if (sessionId == null || sessionId.isBlank()) {
            sessionId = UUID.randomUUID().toString();
        }

        ChatEngine.Result result = engine.getResponse(userMessage == null ? "" : userMessage, sessionId);

        StringBuilder json = new StringBuilder();
        json.append("{\"reply\":\"").append(JsonUtil.escape(result.reply)).append("\",");
        json.append("\"session\":\"").append(JsonUtil.escape(sessionId)).append("\",");
        json.append("\"suggestions\":[");
        for (int i = 0; i < result.suggestions.size(); i++) {
            if (i > 0) json.append(",");
            json.append("\"").append(JsonUtil.escape(result.suggestions.get(i))).append("\"");
        }
        json.append("]}");

        byte[] bytes = json.toString().getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", "application/json; charset=UTF-8");
        exchange.sendResponseHeaders(200, bytes.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }

    // ---------------- Lightweight JSON Utility (no external libraries) ----------------

    static class JsonUtil {
        static String extractValue(String json, String key) {
            String search = "\"" + key + "\"";
            int idx = json.indexOf(search);
            if (idx == -1) return null;
            int colon = json.indexOf(':', idx + search.length());
            if (colon == -1) return null;
            int i = colon + 1;
            while (i < json.length() && Character.isWhitespace(json.charAt(i))) i++;
            if (i >= json.length() || json.charAt(i) != '"') return null;
            i++;
            StringBuilder sb = new StringBuilder();
            while (i < json.length()) {
                char c = json.charAt(i);
                if (c == '\\' && i + 1 < json.length()) {
                    char next = json.charAt(i + 1);
                    switch (next) {
                        case 'n' -> sb.append('\n');
                        case 't' -> sb.append('\t');
                        case '"' -> sb.append('"');
                        case '\\' -> sb.append('\\');
                        default -> sb.append(next);
                    }
                    i += 2;
                } else if (c == '"') {
                    break;
                } else {
                    sb.append(c);
                    i++;
                }
            }
            return sb.toString();
        }

        static String escape(String s) {
            StringBuilder sb = new StringBuilder();
            for (char c : s.toCharArray()) {
                switch (c) {
                    case '"' -> sb.append("\\\"");
                    case '\\' -> sb.append("\\\\");
                    case '\n' -> sb.append("\\n");
                    case '\r' -> sb.append("\\r");
                    case '\t' -> sb.append("\\t");
                    default -> {
                        if (c < 0x20) {
                            sb.append(String.format("\\u%04x", (int) c));
                        } else {
                            sb.append(c);
                        }
                    }
                }
            }
            return sb.toString();
        }
    }

    // ---------------- Chat Engine: NLP preprocessing + weighted scoring + context ----------------

    static class Intent {
        final String tag;
        final String displayName;
        final List<String> responses;
        final List<Set<String>> patternTokens;

        Intent(String tag, String displayName, List<String> patterns, List<String> responses) {
            this.tag = tag;
            this.displayName = displayName;
            this.responses = responses;
            this.patternTokens = new ArrayList<>();
            for (String p : patterns) {
                patternTokens.add(ChatEngine.normalize(p));
            }
        }
    }

    static class ChatEngine {

        static class Result {
            final String reply;
            final List<String> suggestions;
            Result(String reply, List<String> suggestions) {
                this.reply = reply;
                this.suggestions = suggestions;
            }
        }

        private static final Set<String> STOP_WORDS = new HashSet<>(Arrays.asList(
                "a", "an", "the", "is", "are", "am", "was", "were", "do", "does", "did",
                "i", "you", "he", "she", "it", "we", "they", "my", "your", "me", "to",
                "of", "in", "on", "at", "for", "with", "about", "please", "can", "could",
                "would", "will", "what", "how", "when", "where", "which", "who", "tell",
                "there", "this", "that", "and", "or"
        ));

        // Maps related words to one canonical term so phrasing differences don't hurt matching
        private static final Map<String, String> SYNONYMS = new HashMap<>();
        static {
            String[][] groups = {
                {"cost", "price", "amount", "fee", "fees", "tuition"},
                {"job", "jobs", "career", "hiring", "placement", "placements"},
                {"dorm", "dormitory", "hostel", "accommodation", "stay"},
                {"program", "programs", "branch", "branches", "course", "courses", "department", "departments"},
                {"join", "apply", "application", "admission", "admissions", "enroll", "enrollment"},
                {"book", "books", "library"},
                {"call", "phone", "email", "contact"},
                {"schedule", "timing", "timings", "hours"}
            };
            for (String[] group : groups) {
                String canonical = group[0];
                for (String word : group) SYNONYMS.put(word, canonical);
            }
        }

        private static final double MATCH_THRESHOLD = 0.28;
        private static final double CONTEXT_THRESHOLD = 0.12;

        private final List<Intent> intents = new ArrayList<>();
        private final Random random = new Random();
        private final Map<String, Double> idf = new HashMap<>();
        private final Map<String, String> lastTopic = new ConcurrentHashMap<>();      // sessionId -> intent tag
        private final Map<String, Integer> turnCount = new ConcurrentHashMap<>();     // sessionId -> messages seen

        ChatEngine() {
            loadKnowledgeBase();
            computeIdf();
        }

        /** Basic stemming: strips common suffixes so "fees"/"fee", "timings"/"timing" etc. match. */
        private static String stem(String word) {
            if (word.length() > 4 && word.endsWith("ies")) return word.substring(0, word.length() - 3) + "y";
            if (word.length() > 4 && word.endsWith("es")) return word.substring(0, word.length() - 2);
            if (word.length() > 3 && word.endsWith("s") && !word.endsWith("ss")) return word.substring(0, word.length() - 1);
            return word;
        }

        /** Full NLP preprocessing pipeline: clean -> tokenize -> stop-word removal -> stem -> synonym mapping. */
        static Set<String> normalize(String text) {
            String cleaned = text.toLowerCase().replaceAll("[^a-z0-9\\s]", " ");
            Set<String> tokens = new HashSet<>();
            for (String w : cleaned.split("\\s+")) {
                if (w.isBlank() || STOP_WORDS.contains(w)) continue;
                String stemmed = stem(w);
                String canonical = SYNONYMS.getOrDefault(stemmed, SYNONYMS.getOrDefault(w, stemmed));
                tokens.add(canonical);
            }
            return tokens;
        }

        /** Computes inverse-document-frequency style weight per token across the whole pattern corpus. */
        private void computeIdf() {
            Map<String, Integer> docFreq = new HashMap<>();
            int totalPatterns = 0;
            for (Intent intent : intents) {
                for (Set<String> tokens : intent.patternTokens) {
                    totalPatterns++;
                    for (String t : tokens) {
                        docFreq.merge(t, 1, Integer::sum);
                    }
                }
            }
            for (Map.Entry<String, Integer> e : docFreq.entrySet()) {
                double weight = Math.log((1.0 + totalPatterns) / (1.0 + e.getValue())) + 1.0;
                idf.put(e.getKey(), weight);
            }
        }

        private double weight(String token) {
            return idf.getOrDefault(token, 1.0);
        }

        /** TF-IDF weighted Jaccard-style similarity - rarer/more distinctive words count more. */
        private double weightedSimilarity(Set<String> a, Set<String> b) {
            if (a.isEmpty() || b.isEmpty()) return 0.0;
            Set<String> intersection = new HashSet<>(a);
            intersection.retainAll(b);
            Set<String> union = new HashSet<>(a);
            union.addAll(b);

            double interWeight = 0, unionWeight = 0;
            for (String t : intersection) interWeight += weight(t);
            for (String t : union) unionWeight += weight(t);
            return unionWeight == 0 ? 0 : interWeight / unionWeight;
        }

        Result getResponse(String userMessage, String sessionId) {
            turnCount.merge(sessionId, 1, Integer::sum);

            if (userMessage == null || userMessage.isBlank()) {
                return new Result("I didn't quite catch that. Could you please rephrase?", List.of());
            }

            Set<String> userTokens = normalize(userMessage);

            // Rank all intents by weighted similarity
            List<Map.Entry<Intent, Double>> ranked = new ArrayList<>();
            for (Intent intent : intents) {
                double best = 0;
                for (Set<String> patTokens : intent.patternTokens) {
                    best = Math.max(best, weightedSimilarity(userTokens, patTokens));
                }
                ranked.add(Map.entry(intent, best));
            }
            ranked.sort((x, y) -> Double.compare(y.getValue(), x.getValue()));

            Intent topIntent = ranked.get(0).getKey();
            double topScore = ranked.get(0).getValue();

            // 1) Confident direct match
            if (topScore >= MATCH_THRESHOLD) {
                lastTopic.put(sessionId, topIntent.tag);
                return new Result(pick(topIntent.responses), List.of());
            }

            // 2) Context follow-up: short message + a recent topic + some weak similarity to it
            String prevTag = lastTopic.get(sessionId);
            if (prevTag != null && userTokens.size() <= 4) {
                Intent prevIntent = intents.stream().filter(i -> i.tag.equals(prevTag)).findFirst().orElse(null);
                if (prevIntent != null) {
                    double ctxScore = 0;
                    for (Set<String> patTokens : prevIntent.patternTokens) {
                        ctxScore = Math.max(ctxScore, weightedSimilarity(userTokens, patTokens));
                    }
                    boolean genericFollowUp = userTokens.isEmpty()
                            || containsAny(userMessage.toLowerCase(), "more", "much", "many", "else", "detail", "and");
                    if (ctxScore >= CONTEXT_THRESHOLD || genericFollowUp) {
                        lastTopic.put(sessionId, prevIntent.tag);
                        String continuity = "Still on " + prevIntent.displayName.toLowerCase() + " - ";
                        return new Result(continuity + pick(prevIntent.responses), List.of());
                    }
                }
            }

            // 3) No confident match - offer smart "Did you mean...?" suggestions from the top-ranked intents
            List<String> suggestions = new ArrayList<>();
            for (Map.Entry<Intent, Double> entry : ranked) {
                if (entry.getValue() > 0.03 && suggestions.size() < 3) {
                    suggestions.add(entry.getKey().displayName);
                }
            }
            if (!suggestions.isEmpty()) {
                return new Result("I'm not fully sure what you mean. Did you mean one of these?", suggestions);
            }
            return new Result(fallback(), defaultSuggestions());
        }

        private boolean containsAny(String text, String... words) {
            for (String w : words) if (text.contains(w)) return true;
            return false;
        }

        private String pick(List<String> options) {
            return options.get(random.nextInt(options.size()));
        }

        private List<String> defaultSuggestions() {
            return List.of("Courses", "Fees", "Placements");
        }

        private String fallback() {
            String[] fallbacks = {
                "I'm not sure I understand yet. Try asking about courses, fees, admissions, placements, hostel, or the library.",
                "Sorry, I don't have an answer for that. You can ask me about admissions, fees, or placements.",
                "Hmm, that's outside what I've been trained on. Try one of the topics below."
            };
            return fallbacks[random.nextInt(fallbacks.length)];
        }

        private void addIntent(String tag, String displayName, List<String> patterns, List<String> responses) {
            intents.add(new Intent(tag, displayName, patterns, responses));
        }

        private void loadKnowledgeBase() {
            addIntent("greeting", "Greeting",
                Arrays.asList("hi", "hello", "hey", "good morning", "good afternoon", "good evening", "hey there", "yo"),
                Arrays.asList("Hello! Welcome to the Campus Assistant. How can I help you today?",
                              "Hi there! Ask me about courses, fees, admissions, or placements."));

            addIntent("how_are_you", "Small talk",
                Arrays.asList("how are you", "how are you doing", "how's it going", "whats up"),
                Arrays.asList("I'm running smoothly, thanks for asking! How can I help you today?",
                              "Doing great and ready to help. What would you like to know?"));

            addIntent("bye", "Goodbye",
                Arrays.asList("bye", "goodbye", "see you", "talk to you later", "exit", "quit"),
                Arrays.asList("Goodbye! Have a great day.", "See you soon! Feel free to come back anytime."));

            addIntent("thanks", "Thanks",
                Arrays.asList("thanks", "thank you", "thanks a lot", "appreciate it"),
                Arrays.asList("You're welcome!", "Happy to help!", "Anytime!"));

            addIntent("bot_identity", "About me",
                Arrays.asList("what is your name", "who are you", "what are you", "who made you", "who created you"),
                Arrays.asList("I'm the Campus Assistant chatbot, built in Java for CodeAlpha Task 3.",
                              "I'm an AI-based helpdesk chatbot created using Java, NLP preprocessing, and weighted similarity matching."));

            addIntent("courses", "Courses",
                Arrays.asList("what courses do you offer", "list of courses", "which programs are available",
                               "tell me about courses", "branches available", "departments"),
                Arrays.asList("We offer B.Tech programs in IT, CSE, ECE, Mechanical, and Civil Engineering.",
                              "Our college offers courses in Information Technology, Computer Science, Electronics, and more."));

            addIntent("admission", "Admissions",
                Arrays.asList("how to apply for admission", "admission process", "how can I get admission",
                               "application procedure", "how to join"),
                Arrays.asList("You can apply online through our admissions portal by submitting your academic records and entrance score.",
                              "Admissions are based on entrance exam rank and counseling. Visit the admissions office for details."));

            addIntent("fees", "Fees",
                Arrays.asList("what is the fee structure", "how much are the fees", "tuition fee details",
                               "college fees", "fee payment"),
                Arrays.asList("The fee structure varies by department. Please check the official website or contact the accounts office for exact figures.",
                              "Fees can be paid online through the student portal each semester."));

            addIntent("placement", "Placements",
                Arrays.asList("tell me about placements", "placement statistics", "companies visiting campus",
                               "job opportunities after graduation", "placement cell"),
                Arrays.asList("Our placement cell works with companies across IT, core engineering, and product-based sectors every year.",
                              "Placement drives are conducted regularly, and the training cell provides aptitude and interview preparation."));

            addIntent("library", "Library",
                Arrays.asList("library timings", "when does the library open", "library facilities", "book borrowing rules"),
                Arrays.asList("The library is open from 9 AM to 6 PM on all working days.",
                              "You can borrow up to 3 books at a time for two weeks from the central library."));

            addIntent("hostel", "Hostel",
                Arrays.asList("hostel facilities", "is hostel available", "hostel fees", "accommodation details"),
                Arrays.asList("Yes, separate hostel facilities are available for boys and girls with mess and Wi-Fi.",
                              "Hostel accommodation can be requested during admission or through the hostel warden's office."));

            addIntent("timings", "Timings",
                Arrays.asList("college timings", "what time does college start", "class schedule", "working hours"),
                Arrays.asList("College hours are generally from 9:00 AM to 4:00 PM, Monday through Saturday.",
                              "Classes typically run from 9 AM to 4 PM with breaks in between."));

            addIntent("contact", "Contact info",
                Arrays.asList("how can I contact the college", "contact details", "phone number", "email address", "reach out"),
                Arrays.asList("You can reach the college office via the contact page on the official website or visit in person during working hours.",
                              "For queries, please email the admin office or call the front desk listed on the website."));

            addIntent("capabilities", "What I can do",
                Arrays.asList("what can you do", "how can you help me", "your features", "what do you know", "help"),
                Arrays.asList("I can answer FAQs about courses, admissions, fees, placements, hostel, and library facilities - and I remember what we were just discussing!",
                              "Ask me about courses, admissions, fees, placements, hostel, library, or timings!"));
        }
    }

    // ---------------- Frontend: HTML + CSS + JS (single-file web interface) ----------------

    private static final String HTML_PAGE = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Campus Assistant - Chatbot</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(135deg, #4f46e5, #7c3aed);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .chat-container {
                width: 100%;
                max-width: 480px;
                background: #ffffff;
                border-radius: 16px;
                box-shadow: 0 20px 50px rgba(0,0,0,0.3);
                overflow: hidden;
                display: flex;
                flex-direction: column;
                height: 660px;
            }
            .chat-header {
                background: linear-gradient(135deg, #4f46e5, #6366f1);
                color: white;
                padding: 18px 20px;
                display: flex;
                align-items: center;
                gap: 12px;
            }
            .chat-header .avatar {
                width: 40px; height: 40px;
                background: rgba(255,255,255,0.2);
                border-radius: 50%;
                display: flex; align-items: center; justify-content: center;
                font-size: 20px;
            }
            .chat-header h1 { font-size: 16px; font-weight: 600; }
            .chat-header p { font-size: 12px; opacity: 0.85; display: flex; align-items: center; gap: 5px; }
            .status-dot {
                width: 7px; height: 7px; background: #4ade80; border-radius: 50%;
                display: inline-block;
            }
            .chat-messages {
                flex: 1;
                padding: 16px;
                overflow-y: auto;
                background: #f4f5fb;
                display: flex;
                flex-direction: column;
                gap: 4px;
            }
            .msg-row { display: flex; flex-direction: column; margin-bottom: 8px; }
            .msg-row.user { align-items: flex-end; }
            .msg-row.bot { align-items: flex-start; }
            .msg {
                max-width: 78%;
                padding: 10px 14px;
                border-radius: 14px;
                font-size: 14px;
                line-height: 1.4;
                animation: fadeIn 0.2s ease-in;
                white-space: pre-wrap;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(4px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .msg.bot {
                background: white;
                border: 1px solid #e5e7eb;
                border-bottom-left-radius: 4px;
            }
            .msg.user {
                background: #4f46e5;
                color: white;
                border-bottom-right-radius: 4px;
            }
            .timestamp {
                font-size: 10px;
                color: #9ca3af;
                margin-top: 3px;
                padding: 0 4px;
            }
            .typing-indicator {
                display: flex;
                gap: 4px;
                padding: 12px 14px;
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 14px;
                border-bottom-left-radius: 4px;
                width: fit-content;
            }
            .typing-indicator span {
                width: 6px; height: 6px;
                background: #9ca3af;
                border-radius: 50%;
                animation: bounce 1.2s infinite;
            }
            .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
            .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
            @keyframes bounce {
                0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
                30% { transform: translateY(-4px); opacity: 1; }
            }
            .chip-row {
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
                margin-top: 6px;
            }
            .chip {
                background: #eef2ff;
                color: #4f46e5;
                border: 1px solid #c7d2fe;
                border-radius: 14px;
                padding: 6px 12px;
                font-size: 12px;
                cursor: pointer;
            }
            .chip:hover { background: #e0e7ff; }
            .chat-input {
                display: flex;
                padding: 14px;
                gap: 8px;
                border-top: 1px solid #e5e7eb;
                background: white;
            }
            .chat-input input {
                flex: 1;
                padding: 12px 14px;
                border: 1px solid #d1d5db;
                border-radius: 24px;
                font-size: 14px;
                outline: none;
            }
            .chat-input input:focus { border-color: #4f46e5; }
            .chat-input button {
                background: #4f46e5;
                color: white;
                border: none;
                border-radius: 50%;
                width: 44px;
                height: 44px;
                font-size: 18px;
                cursor: pointer;
                transition: background 0.2s;
            }
            .chat-input button:hover { background: #4338ca; }
            .suggestions {
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
                padding: 0 14px 12px 14px;
                background: white;
            }
            .suggestions button {
                background: #eef2ff;
                color: #4f46e5;
                border: none;
                border-radius: 14px;
                padding: 6px 12px;
                font-size: 12px;
                cursor: pointer;
            }
            .suggestions button:hover { background: #e0e7ff; }
        </style>
        </head>
        <body>

        <div class="chat-container">
            <div class="chat-header">
                <div class="avatar">🤖</div>
                <div>
                    <h1>AI Campus Assistant</h1>
                    <p><span class="status-dot"></span>Online • Java NLP Chatbot • CodeAlpha Task 3</p>
                </div>
            </div>

            <div class="chat-messages" id="chatMessages">
                <div class="msg-row bot">
                    <div class="msg bot">Hi! I'm your Campus Assistant. Ask me about courses, admissions, fees, placements, hostel, or library timings - and I'll remember what we're discussing as we chat.</div>
                </div>
            </div>

            <div class="suggestions">
                <button onclick="sendSuggestion('Tell me about courses')">Courses</button>
                <button onclick="sendSuggestion('What are the fees?')">Fees</button>
                <button onclick="sendSuggestion('Tell me about placements')">Placements</button>
                <button onclick="sendSuggestion('Library timings')">Library</button>
            </div>

            <div class="chat-input">
                <input type="text" id="userInput" placeholder="Type your question..." autocomplete="off">
                <button onclick="sendMessage()">➤</button>
            </div>
        </div>

        <script>
            const messagesDiv = document.getElementById('chatMessages');
            const input = document.getElementById('userInput');
            let sessionId = localStorage.getItem('chatSessionId') || '';

            input.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') sendMessage();
            });

            function timeNow() {
                const d = new Date();
                let h = d.getHours(), m = d.getMinutes();
                const ampm = h >= 12 ? 'PM' : 'AM';
                h = h % 12 || 12;
                m = m < 10 ? '0' + m : m;
                return h + ':' + m + ' ' + ampm;
            }

            function appendMessage(text, sender, suggestions) {
                const row = document.createElement('div');
                row.className = 'msg-row ' + sender;

                const bubble = document.createElement('div');
                bubble.className = 'msg ' + sender;
                bubble.textContent = text;
                row.appendChild(bubble);

                if (suggestions && suggestions.length > 0) {
                    const chipRow = document.createElement('div');
                    chipRow.className = 'chip-row';
                    suggestions.forEach(function(s) {
                        const chip = document.createElement('button');
                        chip.className = 'chip';
                        chip.textContent = s;
                        chip.onclick = function() { sendSuggestion('Tell me about ' + s); };
                        chipRow.appendChild(chip);
                    });
                    row.appendChild(chipRow);
                }

                const time = document.createElement('div');
                time.className = 'timestamp';
                time.textContent = timeNow();
                row.appendChild(time);

                messagesDiv.appendChild(row);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            function showTyping() {
                const row = document.createElement('div');
                row.className = 'msg-row bot';
                row.id = 'typingRow';
                row.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
                messagesDiv.appendChild(row);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            function hideTyping() {
                const row = document.getElementById('typingRow');
                if (row) row.remove();
            }

            function sendSuggestion(text) {
                input.value = text;
                sendMessage();
            }

            async function sendMessage() {
                const text = input.value.trim();
                if (!text) return;
                appendMessage(text, 'user');
                input.value = '';
                showTyping();

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: text, session: sessionId })
                    });
                    const data = await response.json();
                    if (data.session) {
                        sessionId = data.session;
                        localStorage.setItem('chatSessionId', sessionId);
                    }
                    hideTyping();
                    appendMessage(data.reply, 'bot', data.suggestions);
                } catch (err) {
                    hideTyping();
                    appendMessage('Sorry, something went wrong connecting to the server.', 'bot');
                }
            }
        </script>

        </body>
        </html>
        """;
}
