import streamlit as st

# Set the title of the app
st.title("Azal Tourism (Turkey)")

# Add an introduction
st.header("Introduction")
st.write("""
Welcome to Azal, a burgeoning travel company with a vision as expansive as the horizons it seeks to explore. At Azal, we believe in more than just travel; we believe in the transformative power of journeys. Founded on the principles of wanderlust, discovery, and connection, Azal beckons travelers to embark on odysseys that transcend the ordinary and redefine the essence of exploration.

As a nascent entity, we are driven by a commitment to excellence, a passion for adventure, and a dedication to crafting unparalleled travel experiences. Join us as we delve into the depths of our vision, mission, and the array of contemporary services we offer to cater to every wanderer's dreams.
""")

# Add vision
st.header("Vision")
st.write("""
At Azal, our vision extends beyond mere travel; it encapsulates a profound desire to inspire a world where wanderlust knows no bounds. We envision a future where every journey is an opportunity for discovery, connection, and transformation. In this world, the desire to explore new destinations, immerse oneself in diverse cultures, and create unforgettable memories becomes ingrained in the fabric of every individual's life.

Our vision is to ignite the spirit of adventure in the hearts of travelers around the globe, fostering a community bound by a shared love for exploration and discovery. We aspire to curate journeys that captivate the imagination, awaken the senses, and leave an indelible mark on the traveler's soul. Our aim is not merely to offer vacations but to orchestrate transformative odysseys that redefine what it means to travel. Through immersive experiences, personalized itineraries, and a deep understanding of our travelers' aspirations, we invite everyone to embark on a voyage of self-discovery, enrichment, and joy. Join us as we aspire to unlock the boundless wonders of the world and create memories that last a lifetime.
""")

# Add mission
st.header("Mission")
st.write("""
At Azal, our mission is clear: to craft extraordinary travel experiences that inspire, enrich, and delight our travelers. We are dedicated to curating personalized journeys that showcase the world's wonders while fostering cultural understanding, environmental stewardship, and sustainable tourism practices. Our commitment to excellence permeates every aspect of our operations, from meticulous travel planning to seamless execution.

With a focus on attention to detail and a passion for exceeding expectations, we strive to provide exceptional service, expert guidance, and unforgettable moments at every step of our travelers' journeys. We believe in the power of travel to create meaningful connections between people and places, transcending boundaries and fostering mutual understanding. Through our dedication to creating memorable experiences, we aim to ignite a lifelong love affair with exploration and leave a positive impact on the communities and destinations we visit. Join us as we embark on a mission to transform dreams into reality and create unforgettable memories that last a lifetime.
""")

# Add information about services
st.header("Services Provided")
services = {
    "Travel Planning and Consultation": "Our experienced travel advisors create personalized itineraries tailored to each traveler's preferences and interests.",
    "Airline Ticket Reservations": "We offer hassle-free airline ticket reservations, ensuring seamless travel arrangements for our clients.",
    "Transportation Services": "Whether it's private transfers, rental cars, or chauffeur services, we provide transportation options to suit every need.",
    "Visa and Passport Assistance": "Our team assists travelers with visa and passport applications, ensuring a smooth and stress-free process.",
    "Travel Insurance": "We offer comprehensive travel insurance coverage to provide peace of mind during your journey.",
    "Tour Packages and Excursions": "From cultural tours to adventure expeditions, we offer a variety of tour packages and excursions to suit every interest.",
    "Cruise Ship Bookings": "Explore the world's oceans with our curated selection of cruise ship bookings.",
    "Event Ticketing": "Whether it's concerts, festivals, or sporting events, we provide ticketing services for a wide range of events around the world.",
    "Travel Documentation": "We handle all travel documentation, including visas, passports, and travel permits, to ensure a seamless travel experience.",
    "Destination Management": "Our destination management services include hotel reservations, restaurant recommendations, and local guides to enhance your travel experience.",
    "Customer Support 24/7": "We provide round-the-clock customer support to address any questions or concerns before, during, and after your travels."
}

for service, description in services.items():
    st.subheader(service)
    st.write(description)

# Add a section for user input
st.header("Learn More About Our Services")
user_query = st.text_input("Ask about our tour offerings, destinations, pricing, reviews, and more:")

if user_query:
    st.write(f"Response for the query '{user_query}':")
    st.write("""
    We provide information about Azal's tour offerings, destinations, pricing, reviews, and other relevant details sourced from their website.
    For more detailed and specific inquiries, please visit our official website or contact our customer support.
    """)

# Add a section for contact information
st.header("Contact Us")
st.write("""
For more information, please visit our official website or contact our customer support:
- **Website:** [Azal Tourism](https://www.azaltourism.com)
- **Email:** info@azaltourism.com
- **Phone:** +90 123 456 7890
""")

# Footer
st.write("© 2024 Azal Tourism. All rights reserved.")
