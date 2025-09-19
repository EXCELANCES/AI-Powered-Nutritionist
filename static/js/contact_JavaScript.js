window.addEventListener("load", function() {
    setTimeout(function() {
        if (document.readyState === 'complete') {
           
            document.querySelector(".logo_container").classList.add("fade-out");
           
            setTimeout(function() {
                document.querySelector(".logo_container").style.display = "none";
                document.querySelector("#main").style.visibility = "visible"
            }, 1000); 
        }
    }, 2000); 

});
// -----------------------------------------------------------------------------------------------
    function setInputEnabled(state) {
        
        document.getElementById('sendButton').disabled = !state;
        if (state) document.getElementById('userQuestion').focus();
    }

    function handleFeedback(type, messageId) {
        fetch('/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                feedback_type: type,
                message_id: messageId
            })
        })
        .then(response => {
            if (!response.ok) throw new Error('Feedback not saved');
            return response.json();
        })
        .then(data => {
            console.log("Feedback saved:", data);
        })
        .catch(error => {
            console.error('Feedback error:', error);
        });
    }



    function askBot() {
        const input = document.getElementById('userQuestion');
        const sendButton = document.getElementById('sendButton');

        const question = input.value.trim();
        if (!question) return;

        setInputEnabled(false); 

    
        appendMessage('user', question);
        input.value = '';

       
        appendMessage('bot', 'Thinking...');

        fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: question })
        })
        .then(response => response.json())
        .then(data => {

            removeLastBotMessage();

            if (data.ai_response) {
                appendMessage('bot', data.ai_response, data.message_id || null);
            }
            if (data.trigger === "nutrition_form") {
                showNutritionForm();  
            }else {
                setInputEnabled(true); 
            }
            
            if (!data.ai_response) {
                appendMessage('bot', 'Sorry, I could not understand that.');
            }
        })
        
        .catch(err => {
        removeLastBotMessage();
        appendMessage('bot', 'An error occurred, please try again.');
        setInputEnabled(true);
        });
    }

    function appendMessage(sender, text, messageId = null, feedbackType = null) {
        const chatMessages = document.getElementById('chat-messages');
        const chatwindow = document.getElementById('chat-window');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

  
        const id = messageId || `msg-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
        messageDiv.dataset.messageId = id;

        if (sender === 'bot') {
            let botMsg = text.replace(/\n/g, "<br>");
            messageDiv.innerHTML = botMsg;

            if (text.trim().toLowerCase() !== "thinking...") {
                const feedbackDiv = document.createElement('div');
                feedbackDiv.className = "feedback-buttons";
                feedbackDiv.style.marginTop = "8px";

                const likeBtn = document.createElement('button');
                likeBtn.innerHTML = "👍";
                likeBtn.title = "Like";
                likeBtn.style.marginRight = "8px";

                const dislikeBtn = document.createElement('button');
                dislikeBtn.innerHTML = "👎";
                dislikeBtn.title = "Dislike";

                
                if (feedbackType === 'like') {
                    likeBtn.classList.add("selected-feedback");
                    likeBtn.disabled = true;
                    dislikeBtn.disabled = true;
                } else if (feedbackType === 'dislike') {
                    dislikeBtn.classList.add("selected-feedback");
                    likeBtn.disabled = true;
                    dislikeBtn.disabled = true;
                }

           
                likeBtn.onclick = function () {
                    handleFeedback('like', id);
                    likeBtn.classList.add("selected-feedback");
                    dislikeBtn.disabled = true;
                    likeBtn.disabled = true;
                };
                dislikeBtn.onclick = function () {
                    handleFeedback('dislike', id);
                    dislikeBtn.classList.add("selected-feedback");
                    likeBtn.disabled = true;
                    dislikeBtn.disabled = true;
                };

                feedbackDiv.appendChild(likeBtn);
                feedbackDiv.appendChild(dislikeBtn);
                messageDiv.appendChild(feedbackDiv);
            }
        } else {
            messageDiv.textContent = text;
        }

        chatMessages.appendChild(messageDiv);
        chatwindow.scrollTop = chatwindow.scrollHeight;
    }

    function removeLastBotMessage() {
        const chatWindow = document.getElementById('chat-window');
        const messages = chatWindow.querySelectorAll('.message.bot');
        if (messages.length > 0) {
            messages[messages.length - 1].remove();
        }
    }

    function showNutritionForm() {
        const chatWindow = document.getElementById('chat-messages');
        const formDiv = document.createElement('div');
        formDiv.className = 'message bot';
        formDiv.id = 'nutrition-form'; 
        formDiv.innerHTML = `
            <div style="padding: 10px;">
                <h3>🥗 Nutritional values:</h3>

                ${createSlider("Calories", 0, 2000, 500)}
                ${createSlider("FatContent", 0, 100, 50)}
                ${createSlider("SaturatedFatContent", 0, 13, 0)}
                ${createSlider("CholesterolContent", 0, 300, 0)}
                ${createSlider("SodiumContent", 0, 2300, 400)}
                ${createSlider("CarbohydrateContent", 0, 325, 100)}
                ${createSlider("FiberContent", 0, 50, 10)}
                ${createSlider("SugarContent", 0, 40, 10)}
                ${createSlider("ProteinContent", 0, 40, 10)}

                <hr style="margin: 20px 0;">
                <h3>⚙️ Recommendation options (OPTIONAL):</h3>

                <label>Number of recommendations: <span id="recCountVal">5</span></label><br>
                <input type="range" min="1" max="20" value="5"
                    oninput="document.getElementById('recCountVal').innerText=this.value"
                    id="RecommendationCount"><br><br>

                <label>Specify ingredients to include (separated by ';'):</label><br>
                <input type="text" id="IncludedIngredients" placeholder="Milk;eggs;butter;chicken..." style="width: 90%; padding: 5px;"><br><br>

                <button id="nutrition-generate-btn" class="nutrition-form-button" onclick="submitNutritionForm()" style="padding: 8px 16px;">Generate</button>
            </div>
        `;
        chatWindow.appendChild(formDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function createSlider(label, min, max, defaultVal) {
        const id = label + "Val";
        return `
            <label>${label}: <span id="${id}">${defaultVal}</span></label><br>
            <input type="range" min="${min}" max="${max}" value="${defaultVal}" 
                oninput="document.getElementById('${id}').innerText=this.value" 
                id="${label}"><br><br>
        `;
    }

    function submitNutritionForm() {
        const oldForm = document.getElementById("nutrition-form");
      
        const generateBtn = document.getElementById("nutrition-generate-btn"); 

       
        if (generateBtn) {
            generateBtn.textContent = "Processing...";
            generateBtn.disabled = true; 
        }
        const data = {
            Calories: parseFloat(document.getElementById("Calories").value),
            Fat: parseFloat(document.getElementById("FatContent").value),
            SaturatedFat: parseFloat(document.getElementById("SaturatedFatContent").value),
            Cholesterol: parseFloat(document.getElementById("CholesterolContent").value),
            Sodium: parseFloat(document.getElementById("SodiumContent").value),
            Carbohydrate: parseFloat(document.getElementById("CarbohydrateContent").value),
            Fiber: parseFloat(document.getElementById("FiberContent").value),
            Sugar: parseFloat(document.getElementById("SugarContent").value),
            Protein: parseFloat(document.getElementById("ProteinContent").value),
            RecommendationCount: parseInt(document.getElementById("RecommendationCount").value),
            IncludedIngredients: document.getElementById("IncludedIngredients").value || ""
        };

       
        fetch("/recommend", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        })
        .then(resp => resp.json())
        .then(result => {
            if (generateBtn) {
                generateBtn.textContent = "Generate";
                generateBtn.disabled = false;
            }
            if (result.recommendations) {
                showRecipeAccordions(result.recommendations);
                if (oldForm) oldForm.remove(); 
            } else {
                appendMessage('bot', "No recommendations found.");
            }
        })
        .catch(err => {
            if (generateBtn) {
                generateBtn.textContent = "Generate";
                generateBtn.disabled = false;
            }
            appendMessage('bot', "An error occurred while generating recommendations.");
            console.error(err);
        });
    }

   function showRecipeAccordions(recipes) {

    
    document.querySelectorAll('.recipe-panel').forEach(panel => panel.remove());

    const chatMessages = document.getElementById('chat-messages');
    const wrapper = document.createElement("div");
    wrapper.className = "message bot recipe-panel";

    const heading = document.createElement("h3");
    heading.textContent = "Recommended recipes:";
    wrapper.appendChild(heading);



    recipes.forEach((recipe, index) => {
        const container = document.createElement("div");
        container.style.border = "1px solid #555";
        container.style.borderRadius = "5px";
        container.style.marginBottom = "10px";
        container.style.overflow = "hidden";

        const header = document.createElement("button");
        header.textContent = recipe.Name;
        header.style.width = "100%";
        header.style.textAlign = "left";
        header.style.padding = "10px";
        header.style.backgroundColor = "#333";
        header.style.color = "#fff";
        header.style.border = "none";
        header.style.cursor = "pointer";
        header.style.fontSize = "16px";
        
        const title = document.createElement("span");
        title.textContent = recipe.Name;
        title.style.flex = "1";
        title.style.fontSize = "16px";
        title.style.textAlign = "left";



        function handleRecipeChoice(recipe,btn) {
            fetch("/log-selection", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(recipe)
            })
            .then(resp => resp.json())
            .then(data => {
                if (data.status === "success") {
                    appendMessage("bot", data.chat_message);  
                    setInputEnabled(true); 
                   
                    if (btn) {
                        btn.textContent = "Selected";
                        btn.style.backgroundColor = "#2ecc40"; 
                        btn.style.border = "1px solid #2ecc40";
                        btn.disabled = true;
                    }
                } else {
                    appendMessage("bot", "Something went wrong while saving your choice.");
                    setInputEnabled(true);
                }
            })
            .catch(err => {
                appendMessage("bot", "Error while saving your selection.");
                setInputEnabled(true); 
                console.error(err);
            });
        }

        // Choose button
        const chooseBtn = document.createElement("button");
        chooseBtn.textContent = "Choose";
        chooseBtn.style.marginLeft = "10px";
        chooseBtn.style.backgroundColor = "#555";
        chooseBtn.style.color = "#fff";
        chooseBtn.style.border = "none";
        chooseBtn.style.padding = "5px 10px";
        chooseBtn.style.cursor = "pointer";
        chooseBtn.style.fontSize = "14px";
        chooseBtn.className = "choose-btn"; 


        chooseBtn.addEventListener("click", (event) => {
            event.stopPropagation();

          
            const parentPanel = chooseBtn.closest('.recipe-panel');
            parentPanel.querySelectorAll('.choose-btn').forEach(b=>{
                b.disabled = true;
                b.style.backgroundColor = "#555";
            });
          
            chooseBtn.textContent = "Selected";
            chooseBtn.style.backgroundColor = "#2ecc40";
            chooseBtn.style.border = "1px solid #2ecc40";

          
            if (!parentPanel.classList.contains('recipe-chosen')) {
                handleRecipeChoice(recipe, chooseBtn);
                parentPanel.classList.add('recipe-chosen');
            }
        });

        header.appendChild(title);
        header.appendChild(chooseBtn);



        const content = document.createElement("div");
        content.style.display = "none";
        content.style.padding = "10px";
        content.style.backgroundColor = "#222";
        content.style.color = "#fff";
        content.innerHTML = buildRecipeHTML(recipe);

        header.addEventListener("click", () => {
            const isOpen = content.classList.contains("accordion-open");
         
            document.querySelectorAll(".accordion-open").forEach(div => {
                div.style.display = "none";
                div.classList.remove("accordion-open");
            });

        
            if (!isOpen) {
                content.style.display = "block";
                content.classList.add("accordion-open");
            }
        });

        container.appendChild(header);
        container.appendChild(content);
        wrapper.appendChild(container);
    });

    chatMessages.appendChild(wrapper);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

    function buildRecipeHTML(recipe) {
        const nutrition = `
            <table border="1" cellpadding="5">
                <tr><th>Calories</th><th>Fat</th><th>Saturated</th><th>Cholesterol</th><th>Sodium</th><th>Carbs</th><th>Fiber</th><th>Sugar</th><th>Protein</th></tr>
                <tr>
                    <td>${recipe.Calories?.toFixed(1)}</td>
                    <td>${recipe.FatContent?.toFixed(1)}</td>
                    <td>${recipe.SaturatedFatContent?.toFixed(1)}</td>
                    <td>${recipe.CholesterolContent?.toFixed(1)}</td>
                    <td>${recipe.SodiumContent?.toFixed(1)}</td>
                    <td>${recipe.CarbohydrateContent?.toFixed(1)}</td>
                    <td>${recipe.FiberContent?.toFixed(1)}</td>
                    <td>${recipe.SugarContent?.toFixed(1)}</td>
                    <td>${recipe.ProteinContent?.toFixed(1)}</td>
                </tr>
            </table>`;

        const ingredients = recipe.RecipeIngredientParts
            ?.split(';')
            .map(i => `<li>${i.trim()}</li>`)
            .join('') || "<li>No ingredients listed</li>";

        const instructions = recipe.RecipeInstructions
            ?.split(';')
            .map(step => `<li>${step.trim()}</li>`)
            .join('') || "<li>No instructions</li>";

        return `
            ${recipe.Images ? `<img src="${recipe.Images}" style="max-width:200px;"><br>` : ""}
            <h5>Nutritional Values (g):</h5>
            ${nutrition}
            <h5>Ingredients:</h5><ul>${ingredients}</ul>
            <h5>Recipe Instructions:</h5><ul>${instructions}</ul>
        `;
    }

 
  document.addEventListener("DOMContentLoaded", function () {
    const input = document.getElementById('userQuestion');
    const form = document.getElementById('profileForm');
    const profileFormContainer = document.getElementById('profile-form-container');
    const chatWindow = document.getElementById('chat-window');
    const inputArea = document.getElementById('input-area');
    const title = document.getElementById('LeTalk');

  
    input.addEventListener('keydown', function (event) {
        if (event.key === 'Enter') {
            var isDisabled = document.getElementById('sendButton').disabled;
            if (isDisabled) {
            
                event.preventDefault();
                return; 
            }
            event.preventDefault();
            askBot();
        }
    });

    // PROFİL FORM SUBMIT
    form.addEventListener('submit', function (e) {
        e.preventDefault();

        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }

        const profileData = {
            name: document.getElementById('name').value,
            age: parseInt(document.getElementById('age').value),
            gender: document.getElementById('gender').value,
            height: parseInt(document.getElementById('height').value),
            weight: parseInt(document.getElementById('weight').value),
            goal: document.getElementById('goal').value,
            activity: document.getElementById('activity').value,
            allergies: document.getElementById('allergies').value.trim().toLowerCase(),
            conditions: document.getElementById('conditions').value.trim().toLowerCase()
        };

        fetch('/submit-profile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(profileData)
        })
        .then(response => response.json())
        .then(data => {
            profileFormContainer.classList.add('fade-out');
            setTimeout(() => {
                profileFormContainer.style.display = 'none';
                chatWindow.style.display = 'block';
                inputArea.style.display = 'flex';
                title.style.display = 'block';
                if (data.chat_message) {
                    appendMessage('bot', data.chat_message, data.message_id || null);
                }
            }, 600);
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    // JSON PROFILE?
    fetch("/static/data/user_profile.json?ts=" + new Date().getTime())
        .then(resp => {
            if (!resp.ok) throw new Error("Profile not found or not readable");
            return resp.json();
        })
        .then(profileData => {
            profileFormContainer.style.display = 'none';
            chatWindow.style.display = 'block';
            inputArea.style.display = 'flex';
            title.style.display = 'block';
          
            fetch('/chat-history')
                .then(res => res.json())
                .then(data => {
                    if (data.chat_history && Array.isArray(data.chat_history)) {
                       
                        data.chat_history.forEach(msg => {
                         
                            if (msg.role === "user" || msg.role === "assistant") {
                                appendMessage(msg.role === "user" ? "user" : "bot", msg.content, msg.id || null, msg.feedback_type || null);
                            }   
                        });
                    }
                })
                .catch(err => {
                    console.warn("Chat geçmişi alınamadı:", err);
                });
        })
        .catch(err => {
            console.warn("No valid profile found. Showing profile form.");
            profileFormContainer.style.display = 'block';
            chatWindow.style.display = 'none';
            inputArea.style.display = 'none';
            title.style.display = 'none';
        });
});


