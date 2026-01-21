"""
Generate training data for voice assistant.
Uses generic placeholders - NO hardcoded menu items.
The intent classifier learns PATTERNS, not specific items.
"""

import csv
import random
from pathlib import Path


def generate_training_data():
    """Generate training data with generic patterns - no hardcoded items"""
    
    data = []
    
    # =====================================================
    # ADD_TO_CART - Adding items to order
    # Pattern: "i want X", "add X", "give me X", etc.
    # =====================================================
    add_to_cart = [
        # Direct orders with "a/an/the/some"
        "i want a",
        "i want the",
        "i want some",
        "i'd like a",
        "i'd like the",
        "i'll have a",
        "i'll have the",
        "i'll take a",
        "i'll take the",
        "can i get a",
        "can i get the",
        "can i have a",
        "can i have the",
        "could i get a",
        "could i have a",
        "give me a",
        "give me the",
        "give me some",
        "get me a",
        "get me the",
        "get me some",
        "add a",
        "add the",
        "add some",
        "order a",
        "order the",
        # With quantities
        "add one",
        "add two",
        "add three",
        "i want one",
        "i want two",
        "i want three",
        "give me one",
        "give me two",
        "give me three",
        "can i get one",
        "can i get two",
        "can i get three",
        "one please",
        "two please",
        "three please",
        "make it one",
        "make it two",
        "make it three",
        # With sizes
        "i want a small",
        "i want a medium",
        "i want a large",
        "make it small",
        "make it medium",
        "make it large",
        "a small one",
        "a medium one",
        "a large one",
        "small please",
        "medium please",
        "large please",
        # Casual speech
        "gimme a",
        "gimme the",
        "gimme some",
        "lemme get a",
        "lemme have a",
        "let me get a",
        "let me have a",
        "throw in a",
        "throw in some",
        "put in a",
        "put in some",
        "hook me up with a",
        "hook me up with some",
        # Confirmations to add
        "yes add it",
        "yes please",
        "add it",
        "add that",
        "put it in",
        "sounds good add it",
        "that one",
        "i'll take that",
        "i'll have that",
        "that sounds good",
    ]
    data.extend([(text, "add_to_cart") for text in add_to_cart])
    
    # =====================================================
    # QUERY_ITEM - Asking about availability/options
    # Pattern: "do you have X?", "is there X?", "what X?"
    # =====================================================
    query_item = [
        # Do you have patterns
        "do you have",
        "do you have any",
        "do you have a",
        "do you have the",
        "do you sell",
        "do you serve",
        "do you offer",
        "do you make",
        # Is there patterns
        "is there",
        "is there a",
        "is there any",
        "are there",
        "are there any",
        # Got any patterns
        "got any",
        "got a",
        "you got",
        "you got any",
        # What do you have patterns
        "what do you have",
        "what do you offer",
        "what do you serve",
        "what options do you have",
        "what choices are there",
        "what kinds do you have",
        "what types do you have",
        # Availability
        "is it available",
        "is that available",
        "are they available",
        "can i get that",
        "can you make that",
        "do you still have",
        "is it on the menu",
        "is that on the menu",
        # Specific queries
        "what about",
        "how about",
        "anything like",
        "something like",
        "anything similar to",
        "options for",
        "choices for",
        # Dietary/special
        "do you have vegetarian",
        "do you have vegan",
        "do you have gluten free",
        "do you have dairy free",
        "do you have sugar free",
        "any vegetarian options",
        "any vegan options",
        "any gluten free options",
        "something without meat",
        "something without dairy",
        "something without gluten",
    ]
    data.extend([(text, "query_item") for text in query_item])
    
    # =====================================================
    # SEARCH_MENU - Browsing/listing menu items
    # Pattern: "show me X", "list X", "what's on menu"
    # =====================================================
    search_menu = [
        # Show me patterns
        "show me",
        "show me the",
        "show me your",
        "show me all",
        "show me what you have",
        "show me options",
        "show me everything",
        # List patterns
        "list",
        "list all",
        "list the",
        "list your",
        "list everything",
        # What's available
        "what's available",
        "what's on the menu",
        "what's on your menu",
        "what do you have",
        "what do you serve",
        "what can i order",
        "what can i get",
        "what are my options",
        "what are the choices",
        # Menu requests
        "menu",
        "menu please",
        "the menu",
        "see the menu",
        "look at the menu",
        "view menu",
        "full menu",
        "entire menu",
        # Category browsing
        "what categories",
        "what sections",
        "show categories",
        "show sections",
        # Recommendations
        "what do you recommend",
        "what's good",
        "what's popular",
        "what's your best",
        "any recommendations",
        "any suggestions",
        "what should i get",
        "what should i order",
    ]
    data.extend([(text, "search_menu") for text in search_menu])
    
    # =====================================================
    # VIEW_MENU - Explicitly viewing full menu
    # Similar to search_menu but for full menu display
    # =====================================================
    view_menu = [
        "view menu",
        "view the menu",
        "see menu",
        "see the menu",
        "open menu",
        "open the menu",
        "display menu",
        "show menu",
        "show full menu",
        "show entire menu",
        "show complete menu",
        "give me the menu",
        "pull up the menu",
        "bring up the menu",
        "let me see the menu",
        "i want to see the menu",
        "can i see the menu",
        "menu view",
    ]
    data.extend([(text, "view_menu") for text in view_menu])
    
    # =====================================================
    # VIEW_CART - Checking current order
    # =====================================================
    view_cart = [
        "what's in my cart",
        "what's in my order",
        "what's in the cart",
        "what did i order",
        "what do i have",
        "what have i ordered",
        "show my cart",
        "show my order",
        "show the cart",
        "view cart",
        "view my cart",
        "view order",
        "view my order",
        "check my cart",
        "check my order",
        "my cart",
        "my order",
        "current order",
        "order so far",
        "what's my order",
        "what's my total",
        "order status",
        "cart status",
        "see my order",
        "see my cart",
        "review order",
        "review my order",
    ]
    data.extend([(text, "view_cart") for text in view_cart])
    
    # =====================================================
    # REMOVE_FROM_CART - Removing items from order
    # =====================================================
    remove_from_cart = [
        "remove",
        "remove the",
        "remove that",
        "remove it",
        "remove one",
        "take off",
        "take off the",
        "take that off",
        "take it off",
        "delete",
        "delete the",
        "delete that",
        "cancel",
        "cancel the",
        "cancel that",
        "i don't want",
        "i don't want the",
        "i don't want that",
        "nevermind the",
        "never mind the",
        "forget the",
        "forget about the",
        "get rid of",
        "get rid of the",
        "no more",
        "actually no",
        "actually remove",
        "scratch that",
        "take it back",
    ]
    data.extend([(text, "remove_from_cart") for text in remove_from_cart])
    
    # =====================================================
    # MODIFY_ITEM - Changing item in cart
    # =====================================================
    modify_item = [
        "change",
        "change the",
        "change that",
        "change it to",
        "modify",
        "modify the",
        "modify that",
        "update",
        "update the",
        "update my order",
        "make it",
        "make that",
        "switch to",
        "switch it to",
        "actually make it",
        "can you change",
        "can you modify",
        "i want to change",
        "i want to modify",
        "change the size",
        "change the quantity",
        "different size",
        "different option",
        "instead of",
        "rather than",
        "swap it for",
        "replace with",
        "replace it with",
    ]
    data.extend([(text, "modify_item") for text in modify_item])
    
    # =====================================================
    # CLEAR_CART - Clearing entire order
    # =====================================================
    clear_cart = [
        "clear cart",
        "clear the cart",
        "clear my cart",
        "clear my order",
        "clear order",
        "clear everything",
        "empty cart",
        "empty the cart",
        "empty my cart",
        "empty my order",
        "start over",
        "start again",
        "start fresh",
        "delete everything",
        "delete all",
        "remove everything",
        "remove all",
        "cancel order",
        "cancel my order",
        "cancel everything",
        "forget everything",
        "nevermind everything",
        "i want to start over",
        "let me start over",
        "reset",
        "reset my order",
        "reset cart",
    ]
    data.extend([(text, "clear_cart") for text in clear_cart])
    
    # =====================================================
    # GET_PRICE - Asking about prices
    # =====================================================
    get_price = [
        "how much",
        "how much is",
        "how much does",
        "how much for",
        "what's the price",
        "what's the cost",
        "what does it cost",
        "what's my total",
        "what's the total",
        "total please",
        "price please",
        "price of",
        "cost of",
        "is it expensive",
        "is that expensive",
        "what do i owe",
        "how much do i owe",
        "the price",
        "the cost",
        "the total",
        "subtotal",
        "grand total",
    ]
    data.extend([(text, "get_price") for text in get_price])
    
    # =====================================================
    # CHECKOUT - Completing order
    # =====================================================
    checkout = [
        "checkout",
        "check out",
        "i'm done",
        "i'm finished",
        "i'm ready",
        "that's all",
        "that's it",
        "that's everything",
        "nothing else",
        "complete order",
        "complete my order",
        "finish order",
        "finish my order",
        "place order",
        "place my order",
        "submit order",
        "submit my order",
        "finalize",
        "finalize order",
        "done ordering",
        "finished ordering",
        "ready to pay",
        "ready to order",
        "let's checkout",
        "proceed to checkout",
        "i want to pay",
        "ring me up",
        "wrap it up",
        "that'll be all",
        "that will be all",
    ]
    data.extend([(text, "checkout") for text in checkout])
    
    # =====================================================
    # CONFIRM_YES - Affirmative responses
    # =====================================================
    confirm_yes = [
        "yes",
        "yeah",
        "yep",
        "yup",
        "yes please",
        "yeah please",
        "sure",
        "sure thing",
        "of course",
        "definitely",
        "absolutely",
        "correct",
        "that's right",
        "that's correct",
        "right",
        "ok",
        "okay",
        "sounds good",
        "perfect",
        "great",
        "go ahead",
        "do it",
        "confirm",
        "confirmed",
        "affirmative",
        "uh huh",
        "mm hmm",
        "for sure",
        "you got it",
        "exactly",
    ]
    data.extend([(text, "confirm_yes") for text in confirm_yes])
    
    # =====================================================
    # CONFIRM_NO - Negative responses
    # =====================================================
    confirm_no = [
        "no",
        "nope",
        "nah",
        "no thanks",
        "no thank you",
        "not really",
        "not yet",
        "not now",
        "i don't think so",
        "actually no",
        "never mind",
        "nevermind",
        "forget it",
        "cancel",
        "stop",
        "wait",
        "hold on",
        "hang on",
        "wrong",
        "that's wrong",
        "that's not right",
        "incorrect",
        "negative",
        "don't",
        "don't do that",
        "uh uh",
        "mm mm",
    ]
    data.extend([(text, "confirm_no") for text in confirm_no])
    
    # =====================================================
    # GREETING - Hello/Hi
    # =====================================================
    greeting = [
        "hi",
        "hello",
        "hey",
        "hey there",
        "hi there",
        "hello there",
        "good morning",
        "good afternoon",
        "good evening",
        "howdy",
        "what's up",
        "sup",
        "yo",
        "greetings",
        "hiya",
    ]
    data.extend([(text, "greeting") for text in greeting])
    
    # =====================================================
    # GOODBYE - Bye/Exit
    # =====================================================
    goodbye = [
        "bye",
        "goodbye",
        "good bye",
        "see you",
        "see ya",
        "later",
        "take care",
        "have a good day",
        "have a nice day",
        "thanks bye",
        "thank you bye",
        "that's all bye",
        "exit",
        "quit",
        "i'm done",
        "gotta go",
        "peace",
        "i'm out",
        "that's all thanks",
        "bye bye",
    ]
    data.extend([(text, "goodbye") for text in goodbye])
    
    # =====================================================
    # HELP - Help requests
    # =====================================================
    help_intent = [
        "help",
        "help me",
        "i need help",
        "what can you do",
        "how does this work",
        "what are my options",
        "i'm confused",
        "i'm lost",
        "show me commands",
        "what can i say",
        "instructions",
        "guide me",
        "how do i order",
        "how do i use this",
        "what should i do",
        "i don't understand",
    ]
    data.extend([(text, "help") for text in help_intent])
    
    return data


def save_training_data(data, output_path):
    """Save training data to CSV"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'intent'])
        for text, intent in data:
            writer.writerow([text, intent])
    
    print(f"✅ Saved {len(data)} examples to {output_path}")


def print_summary(data):
    """Print summary of training data"""
    from collections import Counter
    intent_counts = Counter(intent for _, intent in data)
    
    print("\n📊 Training Data Summary:")
    print("-" * 40)
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent:20} {count:4} examples")
    print("-" * 40)
    print(f"  {'TOTAL':20} {len(data):4} examples")
    print(f"  {'Intents':20} {len(intent_counts):4}")


if __name__ == "__main__":
    data = generate_training_data()
    
    # Shuffle
    random.seed(42)
    random.shuffle(data)
    
    # Print summary
    print_summary(data)
    
    # Save
    output_path = Path("data/training_data_v2.csv")
    output_path.parent.mkdir(exist_ok=True)
    save_training_data(data, output_path)
    
    print(f"\n📁 Output: {output_path}")
    print("\nTo train new model, run:")
    print("  python train_two_models.py")
