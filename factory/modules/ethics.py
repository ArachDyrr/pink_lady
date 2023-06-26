# import unittest
# from unittest.mock import patch
# import io

# def ethics(filosopher="Immanuel Kant"):
#     ontology = f"Don't quote {filosopher} on AI ethics"
#     print(ontology)


# class EthicsTestCase(unittest.TestCase):
#     @patch('sys.stdout', new_callable=io.StringIO)
#     def test_ethics(self, mock_stdout):
#         ethics()
#         self.assertEqual(mock_stdout.getvalue().strip(), "Don't quote Immanuel Kant on AI ethics")

# # Run the test
# if __name__ == '__main__':
#     unittest.main()

# --------------------------------------------------
import logging
import webbrowser

def access_www(url):
    if not url.startswith("https://"):
        logging.warning("Please use https:// in the URL.")
    else:
        webbrowser.open(url)

def ethics(philosopher="Immanuel Kant"):
    dead_philosophers = ["Immanuel Kant", "Plato", "Aristotle", "René Descartes", "Friedrich Nietzsche"]
    if philosopher in dead_philosophers:
        logging.warning(f"{philosopher} is dead.")
    else:
        website_url = "https://plato.stanford.edu/entries/artificial-intelligence/"
        access_www(website_url)

def main():
    logging.basicConfig(level=logging.INFO)
    ethics()

if __name__ == '__main__':
    main()