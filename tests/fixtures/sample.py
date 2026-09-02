class PaymentProcessor:
    def process_payment(self, amount):
        print(f"Processing ${amount}")

def checkout():
    processor = PaymentProcessor()
    processor.process_payment(100)