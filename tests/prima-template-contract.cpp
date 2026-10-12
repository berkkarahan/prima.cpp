#include "prima-chat.h"
#include <fstream>
#include <future>
#include <iostream>
#include <vector>

using json = nlohmann::ordered_json;

int main(int argc, char ** argv) {
    try {
        if (argc == 3) {
            json body;
            std::ifstream(argv[2]) >> body;
            const auto result = prima_render_jinja(nullptr, prima_read_template(argv[1]), body.at("messages"),
                body.value("add_generation_prompt", true), body.value("chat_template_kwargs", json::object()),
                body.value("tools", json()));
            std::cout << json({{"prompt", result}}).dump() << '\n';
            return 0;
        }
        const std::string source = "{% for m in messages %}{{ m.content }}{% endfor %}:{{ label }}";
        std::vector<std::future<bool>> jobs;
        for (int i = 0; i < 8; ++i) jobs.push_back(std::async(std::launch::async, [&, i]() {
            const auto label = std::to_string(i);
            const json messages = json::array({{{"role", "user"}, {"content", "hello"}}});
            for (int j = 0; j < 32; ++j)
                if (prima_render_jinja(nullptr, source, messages, true, {{"label", label}}, nullptr) != "hello:" + label)
                    return false;
            return true;
        }));
        for (auto & job : jobs) if (!job.get()) return 1;
        std::cout << "256 concurrent template contexts verified\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
