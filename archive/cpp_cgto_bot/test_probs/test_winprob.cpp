// Test program to calculate win probabilities for comparison with Python version
// Compile: g++ -std=c++17 -I. -I./include -I./libs/skeleton/include test_winprob.cpp -o test_winprob
// Run: ./test_winprob

#include <bitmask_tables.h>
#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <map>

// Copy Bot struct from main.cpp for testing
struct HandRank {
  std::vector<int> rankTuple;
  int category;

  HandRank(std::vector<int> tuple, int cat) : rankTuple(std::move(tuple)), category(cat) {}

  bool operator>(const HandRank& other) const {
    // Python compares tuples lexicographically: compare elements first
    size_t minSize = std::min(rankTuple.size(), other.rankTuple.size());
    for (size_t i = 0; i < minSize; ++i) {
      if (rankTuple[i] != other.rankTuple[i]) {
        return rankTuple[i] > other.rankTuple[i];
      }
    }
    // If all compared elements are equal, longer tuple wins (in Python, shorter tuple is less)
    return rankTuple.size() > other.rankTuple.size();
  }

  bool operator==(const HandRank& other) const {
    if (rankTuple.size() != other.rankTuple.size()) {
      return false;
    }
    for (size_t i = 0; i < rankTuple.size(); ++i) {
      if (rankTuple[i] != other.rankTuple[i]) {
        return false;
      }
    }
    return true;
  }
};

struct TestBot {
  static constexpr int MC_ITERATIONS = 100;
  static constexpr const char* SUITS = "hdcs";
  
  std::vector<std::string> REMAINING_DECK;
  std::random_device rd;
  std::mt19937 gen;

  TestBot() : gen(rd()) {
  }

  int bitCount(unsigned int x) const {
    int count = 0;
    while (x) {
      count += x & 1;
      x >>= 1;
    }
    return count;
  }

  int getSuitIndex(char suit) const {
    switch (suit) {
      case 'h': return 0;
      case 'd': return 1;
      case 'c': return 2;
      case 's': return 3;
      default: return 0;
    }
  }

  HandRank bestHandRank8(const std::vector<std::string>& cards) const {
    std::array<int, 13> rankCounts = {0};
    std::array<int, 4> suitMasks = {0};
    int rankMask = 0;

    for (const auto& card : cards) {
      char rank = card[0];
      char suit = card[1];
      int rankIdx = RANK_TO_INDEX.at(rank);
      int suitIdx = getSuitIndex(suit);

      rankCounts[rankIdx]++;
      suitMasks[suitIdx] |= (1 << rankIdx);
      rankMask |= (1 << rankIdx);
    }

    // STRAIGHT FLUSH
    for (int suit = 0; suit < 4; ++suit) {
      int sm = suitMasks[suit];
      if (bitCount(sm) >= 5) {
        for (const auto& [mask, high] : STRAIGHT_MASKS) {
          if ((sm & mask) == mask) {
            return HandRank({8, high}, 8);
          }
        }
      }
    }

    // FOUR OF A KIND
    int quad = -1;
    int kicker = -1;
    for (int r = 12; r >= 0; --r) {
      if (rankCounts[r] == 4) {
        quad = r;
      } else if (rankCounts[r] > 0 && kicker == -1) {
        kicker = r;
      }
    }
    if (quad != -1) {
      return HandRank({7, quad, kicker}, 7);
    }

    // FULL HOUSE
    std::vector<int> trips;
    std::vector<int> pairs;
    for (int r = 12; r >= 0; --r) {
      if (rankCounts[r] >= 3) {
        trips.push_back(r);
      } else if (rankCounts[r] >= 2) {
        pairs.push_back(r);
      }
    }
    if (!trips.empty()) {
      int trip = trips[0];
      int pair = (trips.size() > 1) ? trips[1] : (pairs.empty() ? -1 : pairs[0]);
      if (pair != -1) {
        return HandRank({6, trip, pair}, 6);
      }
    }

    // FLUSH
    for (int suit = 0; suit < 4; ++suit) {
      int sm = suitMasks[suit];
      if (bitCount(sm) >= 5) {
        std::vector<int> kickers;
        for (int r = 12; r >= 0; --r) {
          if (sm & (1 << r)) {
            kickers.push_back(r);
          }
        }
        std::vector<int> rankTuple = {5};
        for (size_t i = 0; i < std::min(kickers.size(), size_t(5)); ++i) {
          rankTuple.push_back(kickers[i]);
        }
        return HandRank(rankTuple, 5);
      }
    }

    // STRAIGHT
    for (const auto& [mask, high] : STRAIGHT_MASKS) {
      if ((rankMask & mask) == mask) {
        return HandRank({4, high}, 4);
      }
    }

    // THREE OF A KIND
    if (!trips.empty()) {
      int trip = trips[0];
      std::vector<int> kickers;
      for (int r = 12; r >= 0; --r) {
        if (rankCounts[r] == 1) {
          kickers.push_back(r);
        }
      }
      std::vector<int> rankTuple = {3, trip};
      for (size_t i = 0; i < std::min(kickers.size(), size_t(2)); ++i) {
        rankTuple.push_back(kickers[i]);
      }
      return HandRank(rankTuple, 3);
    }

    // TWO PAIR
    if (pairs.size() >= 2) {
      int highPair = pairs[0];
      int lowPair = pairs[1];
      std::vector<int> kickers;
      for (int r = 12; r >= 0; --r) {
        if (rankCounts[r] == 1) {
          kickers.push_back(r);
        }
      }
      int kicker = kickers.empty() ? -1 : *std::max_element(kickers.begin(), kickers.end());
      return HandRank({2, highPair, lowPair, kicker}, 2);
    }

    // ONE PAIR
    if (!pairs.empty()) {
      int pair = pairs[0];
      std::vector<int> kickers;
      for (int r = 12; r >= 0; --r) {
        if (rankCounts[r] == 1) {
          kickers.push_back(r);
        }
      }
      std::vector<int> rankTuple = {1, pair};
      for (size_t i = 0; i < std::min(kickers.size(), size_t(3)); ++i) {
        rankTuple.push_back(kickers[i]);
      }
      return HandRank(rankTuple, 1);
    }

    // HIGH CARD
    std::vector<int> kickers;
    for (int r = 12; r >= 0; --r) {
      if (rankCounts[r] > 0) {
        kickers.push_back(r);
      }
    }
    std::vector<int> rankTuple = {0};
    for (size_t i = 0; i < std::min(kickers.size(), size_t(5)); ++i) {
      rankTuple.push_back(kickers[i]);
    }
    return HandRank(rankTuple, 0);
  }

  std::pair<double, int> compareHands(const std::vector<std::string>& p1Cards,
                                      const std::vector<std::string>& p2Cards) const {
    HandRank p1Rank = bestHandRank8(p1Cards);
    HandRank p2Rank = bestHandRank8(p2Cards);

    double value;
    if (p1Rank > p2Rank) {
      value = 1.0;
    } else if (p1Rank == p2Rank) {
      value = 0.5;
    } else {
      value = 0.0;
    }
    return {value, p2Rank.category};
  }

  std::pair<double, int> mcOnce(const std::vector<std::string>& myCards,
                                 const std::vector<std::string>& boardCards,
                                 int discardIdx) {
    std::vector<std::string> newBoard = boardCards;
    std::vector<std::string> keptCards = myCards;

    if (discardIdx != -1 && discardIdx < (int)myCards.size()) {
      keptCards.clear();
      for (size_t i = 0; i < myCards.size(); ++i) {
        if (i != (size_t)discardIdx) {
          keptCards.push_back(myCards[i]);
        }
      }
      newBoard = boardCards;
      newBoard.push_back(myCards[discardIdx]);
    }

    std::vector<std::string> deck = REMAINING_DECK;
    std::shuffle(deck.begin(), deck.end(), gen);

    std::vector<std::string> oppCards;
    if (deck.size() >= 2) {
      oppCards = {deck[0], deck[1]};
    }

    int needed = 6 - (int)newBoard.size();
    std::vector<std::string> futureBoard;
    if (deck.size() >= 2 + needed) {
      futureBoard = std::vector<std::string>(deck.begin() + 2, deck.begin() + 2 + needed);
    }

    std::vector<std::string> myHand = keptCards;
    myHand.insert(myHand.end(), newBoard.begin(), newBoard.end());
    myHand.insert(myHand.end(), futureBoard.begin(), futureBoard.end());

    std::vector<std::string> oppHand = oppCards;
    oppHand.insert(oppHand.end(), newBoard.begin(), newBoard.end());
    oppHand.insert(oppHand.end(), futureBoard.begin(), futureBoard.end());

    return compareHands(myHand, oppHand);
  }

  double calcWinningProb(const std::vector<std::string>& myCards,
                        const std::vector<std::string>& boardCards,
                        int street) {
    double wins = 0;
    int total = 0;

    for (int iter = 0; iter < MC_ITERATIONS; ++iter) {
      auto [value, rank] = mcOnce(myCards, boardCards, -1);
      if (rank != 0 || street <= 3) {
        wins += value;
        total += 1;
      }
    }

    total = (street <= 3) ? MC_ITERATIONS : total;
    return wins / MC_ITERATIONS;
  }
};

int main() {
  TestBot bot;
  
  struct TestCase {
    std::vector<std::string> myCards;
    std::vector<std::string> boardCards;
    int street;
    std::string description;
  };
  
  std::vector<TestCase> testCases = {
    {{"Ah", "Kh", "Qh"}, {}, 0, "Pre-flop with premium hand"},
    {{"Ah", "Kh", "Qh"}, {"Jh", "Th"}, 2, "Flop with straight flush draw"},
    {{"As", "Ks", "Qs"}, {"Ac", "Kc", "Qc", "Jc"}, 5, "Turn with two pair"},
    {{"2h", "3h", "4h"}, {"5h", "6h"}, 2, "Flop with low straight"},
    {{"Ah", "Ad", "As"}, {}, 0, "Pre-flop with three aces"},
  };
  
  std::cout << "C++ Win Probability Test Results:" << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  
  for (const auto& test : testCases) {
    // Initialize deck
    bot.REMAINING_DECK.clear();
    for (const char* r = RANKS; *r; ++r) {
      for (const char* s = bot.SUITS; *s; ++s) {
        std::string card;
        card += *r;
        card += *s;
        bool isMyCard = false;
        for (const auto& c : test.myCards) {
          if (c == card) {
            isMyCard = true;
            break;
          }
        }
        if (!isMyCard) {
          bot.REMAINING_DECK.push_back(card);
        }
      }
    }
    
    // Remove board cards
    for (const auto& card : test.boardCards) {
      bot.REMAINING_DECK.erase(
        std::remove(bot.REMAINING_DECK.begin(), bot.REMAINING_DECK.end(), card),
        bot.REMAINING_DECK.end()
      );
    }
    
    double winProb = bot.calcWinningProb(test.myCards, test.boardCards, test.street);
    
    std::cout << test.description << std::endl;
    std::cout << "  Cards: [";
    for (size_t i = 0; i < test.myCards.size(); ++i) {
      std::cout << test.myCards[i];
      if (i < test.myCards.size() - 1) std::cout << " ";
    }
    std::cout << "], Board: [";
    for (size_t i = 0; i < test.boardCards.size(); ++i) {
      std::cout << test.boardCards[i];
      if (i < test.boardCards.size() - 1) std::cout << " ";
    }
    std::cout << "], Street: " << test.street << std::endl;
    std::cout << "  Win probability: " << std::fixed << std::setprecision(6) 
              << winProb << std::endl << std::endl;
  }
  
  std::cout << std::string(80, '=') << std::endl;
  
  return 0;
}

