import scrapy
from scrapy.spiders import CrawlSpider
from scrapy import Request
from copy import deepcopy
from collections import defaultdict


class TripadvisorSpider(CrawlSpider):
    # name of spider
    name = 'tripadvisor'
    # list of allowed domains
    allowed_domains = ['tripadvisor.com']
    # starting url
    start_urls = ['https://www.tripadvisor.com/Hotels-g60898-Atlanta_Georgia-Hotels.html']
    # location of csv file
    custom_settings = {'FEED_URI': 'tmp/tripadvisor.csv'}
    hotel_rating_counter = defaultdict(int)
    base_url = 'https://www.tripadvisor.com'

    def parse(self, response):
        hotels = response.css('.listItem .listing')
        for hotel in hotels:
            rating = float(hotel.css('.info-col span.ui_bubble_rating::attr(alt)').re_first('(\d.?\d?) of 5 bubbles'))
            if self.hotel_rating_counter[rating] > 2:
                continue

            self.hotel_rating_counter[rating] += 1
            link = response.urljoin(hotel.css('.listing_title a.property_title::attr(href)').extract_first())
            yield Request(url=link, callback=self.parse_reviews)

        if response.css('a.next'):
            next_link = response.urljoin(response.css('a.next::attr(href)').extract_first())
            yield Request(url=next_link, callback=self.parse)

    def parse_reviews(self, response):
        name = response.css('.hotelDescription h1::text').extract_first()
        rating = response.css('.rating .ui_bubble_rating::attr(alt)').re_first('(\d.?\d?) of 5 bubbles')
        hotel_dict = {'Name': name, 'Hotel Rating': rating}
        reviews = response.css('.reviewSelector')
        for review in reviews:
            full_review_link = response.urljoin(review.css('.quote a::attr(href)').extract_first())
            meta = deepcopy(response.meta)
            meta['hotel_dict'] = hotel_dict
            yield Request(full_review_link, meta=meta, callback=self.parse_fullreview)

        if response.css('.ui_pagination .next'):
            next_link = response.urljoin(response.css('.ui_pagination .next::attr(href)').extract_first())
            yield scrapy.Request(url=next_link, meta=meta, callback=self.parse_reviews)

    def parse_fullreview(self, response):
        hotel_dict = response.meta.get('hotel_dict')

        member_name = response.css('.member_info .info_text > div::text').extract_first()
        hotel_dict['Member Name'] = member_name
        hotel_dict['Member Location'] = response.css('.member_info .userLoc ::text').extract_first()

        credibility = response.css('.memberOverlayLink .memberBadgingNoText span.badgetext::text').extract()
        if len(credibility) > 1:
            hotel_dict['Number of Contributions'] = credibility[0]
            hotel_dict['Likes on Review'] = credibility[1]
        elif len(credibility) == 1:
            hotel_dict['Number of Contributions'] = credibility[0]

        hotel_dict['Review Rating'] = response.css('span.ui_bubble_rating').xpath('@class').re_first('bubble_(\d)')
        hotel_dict['Review Date'] = response.css('.ratingDate::attr(title)').extract_first()
        review_heading = response.css('.quote .noQuotes::text').extract_first()
        hotel_dict['Review Heading'] = review_heading

        actual_review = response.css('span.fullText ::text').extract_first()
        hotel_dict['Review Text'] = actual_review
        hotel_dict['Review URL'] = response.url
        yield hotel_dict
