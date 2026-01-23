"""
Speech Formatter - Convert text responses to natural, human-like speech
Removes symbols, formats numbers, adds natural pauses, and makes output conversational
"""

import re
import random
from typing import Dict, List


class SpeechFormatter:
    """Format text responses for natural, human-like speech output"""
    
    def __init__(self):
        # Currency symbols
        self.currency_map = {
            '$': 'dollars',
            '€': 'euros',
            '£': 'pounds',
            '¥': 'yen',
        }
        
        # Common abbreviations
        self.abbreviations = {
            'oz': 'ounce',
            'lb': 'pound',
            'kg': 'kilogram',
            'ml': 'milliliter',
            'etc': 'etcetera',
        }
        
        # Natural filler phrases for variety
        self.affirmations = ["Sure!", "Absolutely!", "Of course!", "Got it!", "Alright!"]
        self.transitions = ["So,", "Alright,", "Okay,", "Now,", ""]
        
    def format_for_speech(self, text: str) -> str:
        """
        Convert text to natural, human-like speech.
        
        Examples:
            "$12.95" → "12 dollars and 95 cents"
            "• Item" → "Item"
            "3 items" → "3 items"
            "Welcome to Demo Restaurant!" → "Hi there! Welcome to Demo Restaurant."
        """
        # Detect help text and route to special handler
        if 'I can help you order' in text or 'Try saying:' in text:
            return self.format_help_for_speech(text)
        
        # Detect ingredient lists and format naturally
        if 'contains:' in text.lower() or 'it has:' in text.lower():
            text = self._format_ingredient_list(text)
        
        # Remove emojis
        text = self._remove_emojis(text)
        
        # Humanize common phrases
        text = self.humanize_response(text)
        
        # Format currency
        text = self._format_currency(text)
        
        # Remove bullet points and list markers
        text = self._remove_list_markers(text)
        
        # Simplify parentheses
        text = self._simplify_parentheses(text)
        
        # Format numbers naturally
        text = self._format_numbers(text)
        
        # Add natural pauses with commas
        text = self._add_natural_pauses(text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix double periods
        text = re.sub(r'\.+', '.', text)
        
        return text
    
    def _add_natural_pauses(self, text: str) -> str:
        """Add commas for natural speech pauses"""
        # Add pause after greeting words
        text = re.sub(r'^(Hello|Hi|Hey|Welcome)\s', r'\1, ', text)
        
        # Add pause before "and" in lists of 3+ items
        # "A, B and C" → "A, B, and C" (Oxford comma for clearer speech)
        text = re.sub(r',\s+(\w+)\s+and\s+', r', \1, and ', text)
        
        return text
    
    def _format_ingredient_list(self, text: str) -> str:
        """Format ingredient lists for natural speech"""
        # "X contains: A, B, C, D, E" → "X has A, B, C, D, and E"
        # "It has: A, B, C" → "It has A, B, and C"
        
        # Replace "contains:" with "has"
        text = text.replace("contains:", "has")
        text = text.replace("It has:", "It's made with")
        
        # For long lists, group ingredients naturally
        # "A, B, C, D, E, and 5 more" → sounds better
        if " and " in text and text.count(",") > 5:
            # Long list - keep "and X more" format
            pass
        
        return text
    
    def format_cart_for_speech(self, cart_text: str) -> str:
        """Format cart display for natural speech"""
        lines = cart_text.split('\n')
        speech_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Handle "You have X items"
            if 'You have' in line and 'for' in line:
                # "You have Cheeseburger for $12.95 including tax."
                parts = line.split('for')
                items_part = parts[0].replace('You have', '').strip()
                price_part = parts[1].replace('including tax', '').strip('.')
                speech_lines.append(f"You have {items_part}. Your total is {self.format_for_speech(price_part)}.")
                continue
            
            # Handle item lines with bullets
            if '•' in line or '-' in line[:2]:
                # "• Coffee (Size: Large) - $5.50"
                line = line.replace('•', '').replace('-', '').strip()
                
                # Extract item name and details
                if '(' in line:
                    item_match = re.match(r'(.+?)\s*\((.+?)\)\s*\$?([\d.]+)', line)
                    if item_match:
                        item_name = item_match.group(1).strip()
                        details = item_match.group(2).strip()
                        price = item_match.group(3).strip()
                        speech_lines.append(f"{item_name} with {details}, {price} dollars")
                        continue
                
                # Simple format: "Item - $price"
                if '$' in line:
                    parts = line.split('$')
                    item_name = parts[0].strip()
                    price = parts[1].strip()
                    speech_lines.append(f"{item_name}, {price} dollars")
                    continue
                
                speech_lines.append(line)
                continue
            
            # Handle totals
            if 'Subtotal:' in line or 'Tax:' in line or 'Total:' in line:
                # "Subtotal: $5.50" → "Subtotal 5 dollars and 50 cents"
                label, amount = line.split(':')
                formatted_amount = self.format_for_speech(amount.strip())
                speech_lines.append(f"{label.strip()}, {formatted_amount}")
                continue
            
            # Default: clean up the line
            speech_lines.append(self.format_for_speech(line))
        
        return '. '.join(speech_lines) + '.'
    
    def format_menu_for_speech(self, menu_text: str) -> str:
        """Format menu listing for natural, conversational speech"""
        text = menu_text
        
        # Handle structured format with categories: "pizza: Margherita; salad: Caesar"
        if ';' in text:
            # Extract intro if present
            match = re.match(r'(.+?(?:include|options)):\s*(.+)', text, re.IGNORECASE)
            if match:
                intro = match.group(1).strip()
                rest = match.group(2).strip()
            else:
                intro = ""
                rest = text
            
            # Parse each category segment
            segments = rest.split(';')
            parts = []
            for segment in segments:
                segment = segment.strip()
                if ':' in segment:
                    cat, items = segment.split(':', 1)
                    cat = cat.strip()
                    items = items.strip()
                    # Remove category word from item names (e.g., "Margherita Pizza" → "Margherita")
                    items = self._remove_category_from_items(items, cat)
                    parts.append(f"For {cat}, we have {items}")
                elif segment:
                    parts.append(segment)
            
            # Join with periods
            text = ". ".join(parts)
            if intro:
                text = f"{intro}. {text}"
        else:
            # Handle simple "Our X include:" format
            match = re.match(r'Our (\w+) include:\s*(.+)', text, re.IGNORECASE)
            if match:
                cat = match.group(1)
                items = match.group(2)
                items = self._remove_category_from_items(items, cat)
                text = f"For {cat}, we have {items}"
            
            # Handle "We have these X:" format
            match = re.match(r'We have these (\w+):\s*(.+)', text, re.IGNORECASE)
            if match:
                cat = match.group(1)
                items = match.group(2)
                items = self._remove_category_from_items(items, cat)
                text = f"For {cat}, we have {items}"
            
            # Handle "Our X options:" format
            match = re.match(r'Our (.+?) options:\s*(.+)', text, re.IGNORECASE)
            if match:
                cat = match.group(1)
                items = match.group(2)
                text = f"For {cat} options, we have {items}"
        
        # Make "X more" sound natural
        text = re.sub(r'and (\d+) more', r'plus \1 more', text)
        
        # Clean up double punctuation
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r',\s*\.', '.', text)
        
        # Add conversational ending if not already a question
        if not text.rstrip().endswith('?') and not text.rstrip().endswith('!'):
            text = text.rstrip('.')
            text += ". What sounds good?"
        
        return self.format_for_speech(text)
    
    def _remove_category_from_items(self, items_text: str, category: str) -> str:
        """Remove category word from item names (e.g., 'Margherita Pizza' → 'Margherita')"""
        # Handle singular/plural
        cat_singular = category.rstrip('s')
        cat_title = cat_singular.title()
        
        # Remove " Pizza", " Burger", " Salad" from item names
        items_text = re.sub(rf'\s+{cat_title}\b', '', items_text)
        items_text = re.sub(rf'\s+{cat_singular}\b', '', items_text, flags=re.IGNORECASE)
        
        return items_text
    
    def format_confirmation_for_speech(self, confirmation_text: str) -> str:
        """Format order confirmation for natural, conversational speech"""
        text = confirmation_text
        
        # Make confirmations more friendly
        text = text.replace('Add it to your order?', 'Want me to add it?')
        text = text.replace('Added', "I've added")
        text = text.replace('Your total is now', 'That brings your total to')
        text = text.replace('Would you like to complete', 'Ready to complete')
        
        # Add variety to confirmations
        if "So that's" in text:
            alternatives = ["Alright, that's", "Okay, that's", "Got it, that's"]
            text = text.replace("So that's", random.choice(alternatives))
        
        return self.format_for_speech(text)
    
    def format_help_for_speech(self, help_text: str) -> str:
        """
        Format help text for natural conversational speech.
        Instead of reading a list, give a brief conversational response.
        """
        # Don't read the whole help list - give a short conversational summary
        conversational_help = [
            "Just tell me what you'd like! You can order food, browse the menu, or ask about options. What sounds good?",
            "I can take your order, show you the menu, or help you find something specific. What are you in the mood for?",
            "Sure! Just tell me what you want to order, or ask me to show you a category like pizza or drinks.",
        ]
        return random.choice(conversational_help)
    
    def humanize_response(self, text: str) -> str:
        """Make common assistant responses sound more human and natural"""
        
        # Humanize greetings
        text = re.sub(r'^Welcome to (.+?)!', r"Hi there! Welcome to \1.", text)
        text = re.sub(r'^Hello! Welcome', r"Hey! Welcome", text)
        text = re.sub(r'^Hello!$', random.choice(["Hey!", "Hi there!", "Hello!"]), text)
        
        # Humanize questions
        text = text.replace('What can I get for you today?', 'What can I get you?')
        text = text.replace('What would you like?', 'What would you like?')
        text = text.replace('What would you like to know more about?', 'Anything catch your eye?')
        
        # Humanize confirmations
        text = text.replace('No problem!', 'No worries!')
        text = text.replace('No problem,', 'No worries,')
        text = re.sub(r"Your cart is empty\.?", "You don't have anything in your cart yet.", text)
        
        # Humanize item additions/removals
        text = re.sub(r'Added (.+?) to your order\.?', r"Got it! \1 is in.", text)
        text = re.sub(r'Removed (.+?) from your order\.?', r"Okay, took out the \1.", text)
        text = re.sub(r'Your total is now (\d)', r'Your total comes to \1', text)
        
        # Humanize errors/not found
        text = re.sub(r"I couldn't find.+?on the menu\.?", "Hmm, I don't see that one on the menu.", text)
        text = text.replace("I couldn't find that exact item.", "Hmm, I don't see that one.")
        text = text.replace("I'm not sure what you mean.", "Sorry, I didn't quite catch that.")
        text = text.replace("Did you mean:", "Did you mean")
        
        # Make checkout more natural
        text = text.replace('Ready to checkout!', "Alright, let's wrap this up!")
        text = text.replace('Would you like to complete this order?', 'Should I place this order for you?')
        
        # Humanize goodbye
        text = text.replace('Goodbye!', 'Take care! See you next time.')
        text = text.replace('Have a nice day!', 'Have a great one!')
        
        return text
    
    def _remove_emojis(self, text: str) -> str:
        """Remove emojis from text"""
        # Remove common emojis used in the assistant
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        
        # Remove specific text emojis
        text = text.replace('🤖', '').replace('👤', '').replace('🎤', '')
        
        return text
    
    def _format_currency(self, text: str) -> str:
        """Format currency for speech"""
        # $12.95 → 12 dollars and 95 cents
        # $5.00 → 5 dollars
        
        def replace_currency(match):
            symbol = match.group(1)
            amount = match.group(2)
            
            currency_name = self.currency_map.get(symbol, 'dollars')
            
            # Split dollars and cents
            if '.' in amount:
                dollars, cents = amount.split('.')
                dollars = int(dollars)
                cents = int(cents)
                
                if cents == 0:
                    return f"{dollars} {currency_name}"
                else:
                    cent_word = "cent" if cents == 1 else "cents"
                    return f"{dollars} {currency_name} and {cents} {cent_word}"
            else:
                return f"{amount} {currency_name}"
        
        # Match currency patterns: $12.95, €10.50, etc.
        pattern = r'([\$€£¥])\s*([\d]+\.?\d*)'
        text = re.sub(pattern, replace_currency, text)
        
        return text
    
    def _remove_list_markers(self, text: str) -> str:
        """Remove bullet points and list markers"""
        # Remove bullets
        text = text.replace('•', '')
        text = text.replace('◦', '')
        text = text.replace('▪', '')
        text = text.replace('▫', '')
        
        # Remove numbered lists at start of line
        text = re.sub(r'^\d+[\.\)]\s*', '', text, flags=re.MULTILINE)
        
        return text
    
    def _simplify_parentheses(self, text: str) -> str:
        """Simplify parenthetical content"""
        # (optional) → optional
        # (Size: Large) → with Large size
        
        def replace_parens(match):
            content = match.group(1).strip()
            
            # Special cases
            if ':' in content:
                # (Size: Large) → with Large size
                label, value = content.split(':', 1)
                return f"with {value.strip()} {label.strip().lower()}"
            
            # (optional) → optional
            if content.lower() in ['optional', 'required']:
                return ''
            
            # Default: keep content without parens
            return content
        
        text = re.sub(r'\(([^)]+)\)', replace_parens, text)
        
        return text
    
    def _format_numbers(self, text: str) -> str:
        """Format numbers for natural speech"""
        # Keep small numbers as-is (1, 2, 3...)
        # Format phone numbers, decimals, etc.
        
        # Format percentages: "8%" → "8 percent"
        text = re.sub(r'(\d+)%', r'\1 percent', text)
        
        return text
    
    def format_option_prompt(self, text: str) -> str:
        """Format option selection prompts"""
        # "What size would you like? (Small, Medium, Large)"
        # → "What size would you like? You can choose Small, Medium, or Large"
        
        match = re.search(r'\(([^)]+)\)', text)
        if match:
            options = match.group(1)
            options_list = [opt.strip() for opt in options.split(',')]
            
            if len(options_list) > 1:
                last_option = options_list[-1]
                other_options = ', '.join(options_list[:-1])
                natural_list = f"You can choose {other_options}, or {last_option}"
                text = text.replace(f"({options})", natural_list)
        
        return self.format_for_speech(text)


def format_response_for_tts(text: str, response_type: str = 'general') -> str:
    """
    Convenience function to format any response for TTS.
    
    Args:
        text: Raw text response from assistant
        response_type: Type of response (general, cart, menu, confirmation, option)
    
    Returns:
        Speech-friendly formatted text
    """
    formatter = SpeechFormatter()
    
    if response_type == 'cart':
        return formatter.format_cart_for_speech(text)
    elif response_type == 'menu':
        return formatter.format_menu_for_speech(text)
    elif response_type == 'confirmation':
        return formatter.format_confirmation_for_speech(text)
    elif response_type == 'option':
        return formatter.format_option_prompt(text)
    else:
        return formatter.format_for_speech(text)
